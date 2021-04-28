#include <gunrock/applications/bc.hxx>

using namespace gunrock;
using namespace memory;

void test_sssp(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // --
  // IO

  std::string filename = argument_array[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;
  csr.from_coo(mm.load(filename));

  // --
  // Build graph

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get()   // values
  );  // supports row_indices and column_offsets (default = nullptr)

  // --
  // Params and memory allocation
  
  vertex_t single_source = 0;
  
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> sigmas(n_vertices);
  thrust::device_vector<weight_t> bc_values(n_vertices);

  // --
  // GPU Run
  
  float gpu_elapsed = gunrock::bc::run(
    G, 
    single_source, 
    sigmas.data().get(), 
    bc_values.data().get()
  );

  // --
  // Log + Validate

  std::cout << "GPU sigmas (output)    = ";
  thrust::copy(sigmas.begin(),
               (sigmas.size() < 40) ? sigmas.begin() + sigmas.size()
                                       : sigmas.begin() + 40,
               std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "GPU bc_values (output) = ";
  thrust::copy(bc_values.begin(),
               (bc_values.size() < 40) ? bc_values.begin() + bc_values.size()
                                       : bc_values.begin() + 40,
               std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
}
