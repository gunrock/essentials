#include <gunrock/algorithms/spmv.hxx>
#include <gunrock/algorithms/generate/random.hxx>
#include <gunrock/graph/reorder.hxx>
#include <sys/time.h>
#include <gunrock/util/timer.hxx>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "spmv_cpu.hxx"

using namespace gunrock;
using namespace memory;

void test_spmv(int num_arguments, char** argument_array) {
  if (num_arguments != 4) {
    std::cerr << "usage: ./bin/<program-name> reorder filename.mtx GS"
              << std::endl;
    exit(1);
  }

  // --
  // Define types
  // Specify the types that will be used for
  // - vertex ids (vertex_t)
  // - edge offsets (edge_t)
  // - edge weights (weight_t)

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // --
  // IO

  gunrock::util::timer_t reorderTimer;
  // Filename to be read
  std::string GS = argument_array[3];
  std::string filename = argument_array[2];
  std::string reorder = argument_array[1];
  // Load the matrix-market dataset into csr format.
  // See `format` to see other supported formats.
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;
  using coo_t =
      format::coo_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  coo_t coo = mm.load(filename);
  // auto cooO = coo;
  float reorder_time = 0;
  util::cpu_timer_t copyTime;
  copyTime.begin();
  coo_t coo2 = coo;
  copyTime.end();
  printf("copy time is %f\n", copyTime.milliseconds());

  if (reorder != "nore") {
    auto context =
        std::shared_ptr<gcuda::multi_context_t>(new gcuda::multi_context_t(0));
    graph::reorder::random(coo2, coo, context);
    if (GS.find("write") != std::string::npos) {
      //  graph::reorder::random(coo2, coo, context);
      graph::reorder::uniquify(coo, coo2,
                               std::shared_ptr<gcuda::multi_context_t>(
                                   new gcuda::multi_context_t(0)));
      using coo_t =
          format::coo_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
      coo_t cooh = coo2;
      graph::reorder::write_mtx(cooh, GS.c_str());
      return;
    }
    if (GS.find("edgeW") != std::string::npos) {
      //  graph::reorder::random(coo2, coo, context);
      using coo_t =
          format::coo_t<memory_space_t::host, vertex_t, edge_t, weight_t>;

      coo_t cooh = coo;
      graph::reorder::edge_order(coo, coo2, cooh,
                                 std::shared_ptr<gcuda::multi_context_t>(
                                     new gcuda::multi_context_t(0)));

      cooh = coo2;
      graph::reorder::write_mtx(cooh, GS.c_str());
      return;
    }
    // auto t1 = getTime();
    reorderTimer.begin();
    if (reorder == "reorder") {
      graph::reorder::uniquify(coo, coo2,
                               std::shared_ptr<gcuda::multi_context_t>(
                                   new gcuda::multi_context_t(0)));
    }
    // graph::reorder::uniquify2(coo, coo2);
    // graph::reorder::random(coo, coo2);
    // graph::reorder::degree(coo, coo2);
    //  auto t2 = getTime();
    reorder_time = reorderTimer.end();
    printf("reorder:%f \n", reorder_time);
  }
  util::cpu_timer_t tt;
  tt.begin();
  if (reorder == "reorder")
    csr.from_coo(coo2);
  else
    csr.from_coo(coo);
  auto buildcsr = tt.end();

  printf("Building CSR:%f \n", buildcsr);

  // --
  // Build graph

  // Convert the dataset you loaded into an `essentials` graph.
  // `memory_space_t::device` -> the graph will be created on the GPU.
  // `graph::view_t::csr`     -> your input data is in `csr` format.
  //
  // Note that `graph::build::from_csr` expects pointers, but the `csr` data
  // arrays are `thrust` vectors, so we need to unwrap them w/ `.data().get()`.
  //  auto b1 = getTime();
  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows, csr.number_of_columns, csr.number_of_nonzeros,
      csr.row_offsets.data().get(), csr.column_indices.data().get(),
      csr.nonzero_values.data().get());

  // --
  // Params and memory allocation
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> x(n_vertices);
  thrust::device_vector<weight_t> y(n_vertices);
  // thrust::device_vector<weight_t> yy(n_vertices);
  // thrust::device_vector<weight_t> w(n_vertices);

  gunrock::generate::random::uniform_distribution(x);
  // auto b2 = getTime();
  // printf("build Graph:%f \n",b2-b1);
  // --
  // GPU Run
  float gpu_elapsed = gunrock::spmv::run(G, x.data().get(), y.data().get());

  // gunrock::print::head(cooO.row_indices, 40, "cooO");
  // gunrock::print::head(coo.row_indices, 40, "coo");
  // gunrock::print::head(coo2.row_indices, 40, "coo2");
  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;

  using json = nlohmann::json;

  auto gs = GS == "GS" ? graph::reorder::gscore(G) : 0;
  printf("GSCORE %i\n", gs);
  std::string fname = reorder + "_" + GS + "_spmv_results.json";
  bool output_file_exist = std::filesystem::exists(std::string("./") + fname);
  std::fstream output(std::string("./") + fname, std::ios::app);

  std::string basename = std::filesystem::path(filename).filename();

  /* FOR TEST
  csr_t csr_T;
  csr_T.from_coo(cooO);

  auto G_T = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr_T.number_of_rows, csr_T.number_of_columns, csr_T.number_of_nonzeros,
      csr_T.row_offsets.data().get(), csr_T.column_indices.data().get(),
      csr_T.nonzero_values.data().get());

  auto exT2 = gunrock::spmv::run(G, x.data().get(), yy.data().get());
  thrust::sort(y.begin(),y.end());
  thrust::sort(yy.begin(),yy.end());
  thrust::transform(y.begin(),y.end(),yy.begin(),w.begin(),thrust::minus<float>());
  printf("TEST: %f, %i, %i, %f \n",
  thrust::reduce(w.begin(),w.end()),G.get_number_of_vertices(),G_T.get_number_of_vertices(),
  exT2); gunrock::print::head(y, 40, "y"); gunrock::print::head(yy, 40, "yy");


  /*if (!output_file_exist) {
    output << "graph_name, M, N, CSR_build_time, Reorder_Time, SPMV_Run, GSCORE,
  \n";
  }
  output << basename << "," << G.get_number_of_vertices() << "," <<
  G.get_number_of_edges() << "," << buildcsr << "," << reorder_time << ","
         << gpu_elapsed << "," << gs << ",\n";
  */
  json record;
  record["graph_name"] = basename;
  record["M"] = G.get_number_of_edges();
  record["N"] = G.get_number_of_vertices();
  record["CSR_Build_Time"] = buildcsr;
  if (reorder == "reorder")
    record["Reorder_Time"] = reorder_time;
  record["SPMV_Run"] = gpu_elapsed;
  if (GS == "GS")
    record["GSCORE"] = gs;
  output << record << "\n";

  // --
  // CPU Run

  thrust::host_vector<weight_t> y_h(n_vertices);
  float cpu_elapsed = spmv_cpu::run(csr, x, y_h);

  // --
  // Log + Validate
  int n_errors = util::compare(
      y.data().get(), y_h.data(), n_vertices,
      [=](const weight_t a, const weight_t b) {
        // TODO: needs better accuracy.
        return std::abs(a - b) > 1e-2;
      },
      true);

  gunrock::print::head(y, 40, "GPU y-vector");
  gunrock::print::head(y_h, 40, "CPU y-vector");

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
  std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  std::cout << "Number of errors : " << n_errors << std::endl;
}

// Main method, wrapping test function
int main(int argc, char** argv) {
  test_spmv(argc, argv);
}