#include <gunrock/algorithms/spmv.hxx>
#include <gunrock/algorithms/generate/random.hxx>
#include <gunrock/graph/reorder.hxx>
#include <sys/time.h>
#include <gunrock/util/timer.hxx>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
using namespace gunrock;
using namespace memory;

/*double getTime() { struct timeval tv; gettimeofday(&tv, 0); return tv.tv_sec *
  1000.0 + tv.tv_usec / 1000.0;
  } */
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
  using csc_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csc;

  using coo_t =
      format::coo_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  coo_t coo = mm.load(filename);
  // auto cooO = coo;
  float reorder_time = 0;
  auto copyTime = getTime();

  coo_t coo2;
  if (reorder != "nore") {
    coo2 = coo;
    auto copyTime2 = getTime();
    printf("copy time is %f\n", copyTime2 - copyTime);

    auto context =
        std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0));
    // graph::reorder::random(coo2, coo, context);

    if (GS.find("write_random") != std::string::npos) {
      graph::reorder::random(coo2, coo, context);
      //      graph::reorder::uniquify(
      //		       coo, coo2,
      //		       std::shared_ptr<cuda::multi_context_t>(new
      // cuda::multi_context_t(0)));
      using coo_t =
          format::coo_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
      coo_t cooh = coo;
      graph::reorder::write_mtx(cooh, GS.c_str());
      return;
    }
    if (GS.find("strided") != std::string::npos) {
      // graph::reorder::random(coo2, coo, context);
      graph::reorder::uniquify_strided(
          coo, coo2,
          std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0)));
      using coo_t =
          format::coo_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
      coo_t cooh = coo2;
      graph::reorder::write_mtx(cooh, GS.c_str());
      return;
    }
    if (GS.find("write_reorder") != std::string::npos) {
      // graph::reorder::random(coo2, coo, context);
      graph::reorder::uniquify(
          coo, coo2,
          std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0)));
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

      coo_t cooh = mm.load(filename);
      graph::reorder::edge_order(
          coo, coo2, cooh,
          std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0)));

      // cooh = coo2;
      // graph::reorder::write_mtx(cooh,GS.c_str());
      return;
    }
    // auto t1 = getTime();
    reorderTimer.begin();
    if (reorder == "reorder") {
      graph::reorder::uniquify(
          coo, coo2,
          std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0)));
      //  graph::reorder::degree(
      //       coo, coo2,
      //       std::shared_ptr<cuda::multi_context_t>(new
      //       cuda::multi_context_t(0)));
    }
    // graph::reorder::uniquify2(coo, coo2);
    // graph::reorder::random(coo, coo2);
    // graph::reorder::degree(coo, coo2);
    //  auto t2 = getTime();
    reorder_time = reorderTimer.end();
    printf("reorder:%f \n", reorder_time);
  }
  auto tt = getTime();
  if (reorder == "reorder")
    csr.from_coo(coo2);
  else
    csr.from_coo(coo);

  // csc.from_coo(coo);

  auto tt2 = getTime();
  auto buildcsr = tt2 - tt;
  printf("Building CSR:%f \n", tt2 - tt);

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

  thrust::device_vector<vertex_t> row_indices(csr.number_of_nonzeros);
  thrust::device_vector<vertex_t> column_indices(csr.number_of_nonzeros);
  thrust::device_vector<edge_t> column_offsets(csr.number_of_columns + 1);

  // --
  // Params and memory allocation
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> x(n_vertices);
  thrust::device_vector<weight_t> y(n_vertices);
  // thrust::device_vector<weight_t> yy(n_vertices);
  //  thrust::device_vector<weight_t> w(n_vertices);

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

  auto cG = graph::build::from_csr<memory_space_t::device, graph::view_t::csc>(
      csr.number_of_rows, csr.number_of_columns, csr.number_of_nonzeros,
      csr.row_offsets.data().get(), csr.column_indices.data().get(),
      csr.nonzero_values.data().get(), row_indices.data().get(),
      column_offsets.data().get());

  // auto gs = 0;
  auto gs = GS == "AID" ? graph::reorder::aid(cG) : 0;
  printf("AID %llu\n", gs);

  auto ggs = GS == "AIDCSR" ? graph::reorder::aidCSR(G) : 0;
  printf("AIDCSR %llu\n", ggs);

  auto sectorCSR = GS == "SECCSR" ? graph::reorder::avgCacheLinesCSR(G, 8) : 0;
  printf("SECCSR %llu\n", sectorCSR);

  auto sectorCSC = GS == "SECCSC" ? graph::reorder::avgCacheLinesCSC(cG, 8) : 0;
  printf("SECCSC %llu\n", sectorCSC);

  auto sectorNbr = GS == "NBR" ? graph::reorder::avgCacheNbr(G, 8) : 0;
  printf("SecNbr %f\n", sectorNbr);

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
  std::cout << record << std::endl;
}

// Main method, wrapping test function
int main(int argc, char** argv) {
  test_spmv(argc, argv);
}