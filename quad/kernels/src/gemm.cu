#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/util/device_memory.h>
#include <gemm.h>

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

void matmul_w4a4_host(
    const Int4Storage *A,
    const Int4Storage *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int32_t *C)
{
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::int4b_t,               // ElementA
        cutlass::layout::RowMajor,      // LayoutA
        cutlass::int4b_t,               // ElementB
        cutlass::layout::ColumnMajor,   // LayoutB
        int32_t,                        // ElementOutput
        cutlass::layout::RowMajor,      // LayoutOutput
        int32_t,                        // ElementAccumulator
        cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
        cutlass::arch::Sm80             // tag indicating target GPU compute architecture  // TODO: This is just for compiling on my laptop temporarily. Should be higher when doing benchmarking.
        >;

    Gemm gemmOp;

    using GemmCoord = cutlass::gemm::GemmCoord;

    typename Gemm::Arguments arguments{
        {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N), static_cast<GemmCoord::Index>(K)},
        {(cutlass::int4b_t *)A, K},
        {(cutlass::int4b_t *)B, K},
        {C, N},
        {C, N},
        {1, 0}};

    auto status = gemmOp(arguments);

    ensure(status == cutlass::Status::kSuccess,
           cutlassGetStatusString(status));
}

void matmul_w8a8_host(
    const int8_t *A,
    const int8_t *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int32_t *C)
{
    using Gemm = cutlass::gemm::device::Gemm<
        int8_t,               // ElementA
        cutlass::layout::RowMajor,      // LayoutA
        int8_t,               // ElementB
        cutlass::layout::ColumnMajor,   // LayoutB
        int32_t,                        // ElementOutput
        cutlass::layout::RowMajor,      // LayoutOutput
        int32_t,                        // ElementAccumulator
        cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
        cutlass::arch::Sm80             // tag indicating target GPU compute architecture  // TODO: This is just for compiling on my laptop temporarily. Should be higher when doing benchmarking.
        >;

    Gemm gemmOp;

    using GemmCoord = cutlass::gemm::GemmCoord;

    typename Gemm::Arguments arguments{
        {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N), static_cast<GemmCoord::Index>(K)},
        {A, K},
        {B, K},
        {C, N},
        {C, N},
        {1, 0}};

    auto status = gemmOp(arguments);

    ensure(status == cutlass::Status::kSuccess,
           cutlassGetStatusString(status));
}


void matmul_w4a8_host(
    const int8_t *A,
    const Int4Storage *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int32_t *C)
{
    using ElementA = int8_t;
    using ElementB = cutlass::int4b_t;
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;

    using DeviceGemmT = cutlass::gemm::device::GemmUniversal<
        ElementA,
        cutlass::layout::RowMajor,
        ElementB,
        cutlass::layout::ColumnMajor,
        ElementOutput,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4,  // Stages
        16, // AlignmentA
        32, // AlignmentB
        cutlass::arch::OpMultiplyAddMixedInputUpcast,
        cutlass::ComplexTransform::kNone,
        cutlass::ComplexTransform::kNone>;

    typename DeviceGemmT::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K}, // problem size
        1,
        {1, 0},
        A,
        B,
        C,
        C,
        M * K,
        N * K,
        M * N,
        M * N, // batch strides
        K, // lda
        K, // ldb
        N, // ldc
        N, // ldd
    };
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = DeviceGemmT::get_workspace_size(arguments);
    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    DeviceGemmT device_gemm;
    // Check the problem size is supported or not
    CUTLASS_CHECK(device_gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(device_gemm());
}
