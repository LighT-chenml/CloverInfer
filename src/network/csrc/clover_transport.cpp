#include <torch/extension.h>
#include <infiniband/verbs.h>
#include <iostream>
#include <vector>
#include <stdexcept>

// Minimal Error Checking Macro
// Minimal Error Checking Macro
#define CHECK_VERBS(stmt) do { \
    int ret = (stmt); \
    if (ret != 0) { \
        std::cerr << "Verbs Error: " << ret << std::endl; \
        throw std::runtime_error("Verbs Error"); \
    } \
} while(0)

class RDMAContext {
public:
    struct ibv_context* ctx = nullptr;
    struct ibv_pd* pd = nullptr;

    RDMAContext(const std::string& dev_name) {
        int num_devices;
        struct ibv_device** device_list = ibv_get_device_list(&num_devices);
        if (!device_list) throw std::runtime_error("Failed to get IB devices");

        for (int i = 0; i < num_devices; ++i) {
            if (std::string(ibv_get_device_name(device_list[i])) == dev_name) {
                ctx = ibv_open_device(device_list[i]);
                break;
            }
        }
        ibv_free_device_list(device_list);

        if (!ctx) throw std::runtime_error("Device not found: " + dev_name);

        pd = ibv_alloc_pd(ctx);
        if (!pd) throw std::runtime_error("Failed to alloc PD");
        
        std::cout << "RDMA Context Initialized on " << dev_name << std::endl;
    }

    ~RDMAContext() {
        if (pd) ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
};

class RDMAEndpoint {
    struct ibv_qp* qp = nullptr;
    struct ibv_cq* cq = nullptr;
    RDMAContext* r_ctx;
    
public:
    struct ConnectionInfo {
        uint32_t qpn;
        uint16_t lid;
    };

    RDMAEndpoint(RDMAContext* r_ctx) : r_ctx(r_ctx) {
        cq = ibv_create_cq(r_ctx->ctx, 128, nullptr, nullptr, 0); 
        if (!cq) throw std::runtime_error("Failed to create CQ");

        struct ibv_qp_init_attr attr = {};
        attr.send_cq = cq;
        attr.recv_cq = cq;
        attr.cap.max_send_wr = 128;
        attr.cap.max_recv_wr = 128;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        attr.qp_type = IBV_QPT_RC;

        qp = ibv_create_qp(r_ctx->pd, &attr);
        if (!qp) throw std::runtime_error("Failed to create QP");

        // INIT State
        struct ibv_qp_attr init_attr = {};
        init_attr.qp_state = IBV_QPS_INIT;
        init_attr.pkey_index = 0;
        init_attr.port_num = 1;
        init_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

        CHECK_VERBS(ibv_modify_qp(qp, &init_attr, 
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
    }

    ConnectionInfo get_info() {
        struct ibv_port_attr port_attr;
        ibv_query_port(r_ctx->ctx, 1, &port_attr);
        return {qp->qp_num, port_attr.lid};
    }

    void connect(uint32_t dest_qpn, uint16_t dest_lid) {
        // RTS Transition
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_4096; // Assumption
        attr.dest_qp_num = dest_qpn;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 0;
        attr.ah_attr.dlid = dest_lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = 1;

        std::cout << "Transitioning to RTR..." << std::endl;
        CHECK_VERBS(ibv_modify_qp(qp, &attr, 
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | 
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));

        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;

        std::cout << "Transitioning to RTS..." << std::endl;
        CHECK_VERBS(ibv_modify_qp(qp, &attr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | 
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
            
        std::cout << "QP Connected to LID " << dest_lid << " QPN " << dest_qpn << std::endl;
    }

    // Register Memory Region
    // Returns a raw pointer (uint64_t) to ibv_mr
    int64_t register_mr(torch::Tensor tensor) {
         void* ptr = tensor.data_ptr();
         size_t size = tensor.numel() * tensor.element_size();
         
         struct ibv_mr* mr = ibv_reg_mr(r_ctx->pd, ptr, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
         if (!mr) throw std::runtime_error("Failed to register MR");
         return (int64_t)mr;
    }

    void post_send(int64_t mr_handle, torch::Tensor tensor) {
        struct ibv_mr* mr = (struct ibv_mr*)mr_handle;
        struct ibv_sge sge;
        sge.addr = (uint64_t)tensor.data_ptr();
        sge.length = tensor.numel() * tensor.element_size();
        sge.lkey = mr->lkey;

        struct ibv_send_wr wr = {};
        wr.wr_id = 1; // Dummy ID
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;

        struct ibv_send_wr* bad_wr;
        CHECK_VERBS(ibv_post_send(qp, &wr, &bad_wr));
    }

    void post_recv(int64_t mr_handle, torch::Tensor tensor) {
        struct ibv_mr* mr = (struct ibv_mr*)mr_handle;
        struct ibv_sge sge;
        sge.addr = (uint64_t)tensor.data_ptr();
        sge.length = tensor.numel() * tensor.element_size();
        sge.lkey = mr->lkey;

        struct ibv_recv_wr wr = {};
        wr.wr_id = 2; // Dummy ID
        wr.sg_list = &sge;
        wr.num_sge = 1;

        struct ibv_recv_wr* bad_wr;
        CHECK_VERBS(ibv_post_recv(qp, &wr, &bad_wr));
    }
    
    // Simple Polling (Blocking for demo, usually Async)
    void poll() {
        struct ibv_wc wc;
        while(ibv_poll_cq(cq, 1, &wc) == 0) {
            // Busy wait
        }
        if (wc.status != IBV_WC_SUCCESS) {
            std::cout << "WC Error: " << wc.status << std::endl;
            throw std::runtime_error("WC Error");
        }
    }
};

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<RDMAContext>(m, "RDMAContext")
        .def(py::init<const std::string&>());

    py::class_<RDMAEndpoint>(m, "RDMAEndpoint")
        .def(py::init<RDMAContext*>())
        .def("get_info", [](RDMAEndpoint& self) {
            auto info = self.get_info();
            return std::make_pair(info.qpn, info.lid);
        })
        .def("connect", &RDMAEndpoint::connect)
        .def("register_mr", &RDMAEndpoint::register_mr)
        .def("post_send", &RDMAEndpoint::post_send)
        .def("post_recv", &RDMAEndpoint::post_recv)
        .def("poll", &RDMAEndpoint::poll);
}
