#pragma once
#include <gunrock/cuda/cuda.hxx>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <stdio.h>
#include <gunrock/formats/coo.hxx>
#include <gunrock/formats/csr.hxx>
#include <thrust/reduce.h>

namespace gunrock {
namespace graph {
namespace reorder {

template <typename coo_host_t>
void write_el(coo_host_t& G, const char* fname) {
  printf("writing %s \n", fname);
  FILE* file;
  int M = G.number_of_nonzeros;
  int N = G.number_of_rows;

  auto I = thrust::raw_pointer_cast(G.row_indices.data());
  auto J = thrust::raw_pointer_cast(G.column_indices.data());

  if (file = fopen(fname, "w")) {
    for (int i = 0; i < M; ++i)
      fprintf(file, "%i %i\n", I[i], J[i]);

    fclose(file);
  }
}
template <typename coo_host_t>
void write_mtx(coo_host_t& G, const char* fname) {
  printf("writing %s \n", fname);
  FILE* file;
  int M = G.number_of_nonzeros;
  int N = G.number_of_rows;

  auto I = thrust::raw_pointer_cast(G.row_indices.data());
  auto J = thrust::raw_pointer_cast(G.column_indices.data());

  if (file = fopen(fname, "w")) {
    fprintf(file, "%%%%MatrixMarket matrix coordinate pattern general \n");
    fprintf(file, "%i %i %i\n", N, N, M);
    for (int i = 0; i < M; ++i)
      fprintf(file, "%i %i\n", I[i] + 1, J[i] + 1);

    fclose(file);
  }
}
template <typename csc_device_t>
__device__ void cacheCSC_score(int i,
                               unsigned long long& score,
                               csc_device_t& G,
                               int stride = 8) {
  auto offsets = G.get_column_offsets();
  auto c2 = offsets[i + 1] / stride;
  auto c1 = offsets[i] / stride;
  score = c2 - c1 + 1;
}

template <typename csc_device_t>
unsigned long long avgCacheLinesCSC(
    csc_device_t& G,
    int stride = 8,
    std::shared_ptr<cuda::multi_context_t> context =
        std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0))) {
  std::shared_ptr<cuda::standard_context_t> scontext =
      std::shared_ptr<cuda::standard_context_t>(context->get_context(0));

  int N = G.get_number_of_vertices();
  thrust::device_vector<unsigned long long> scores(N, 0);
  unsigned long long* pcache = thrust::raw_pointer_cast(scores.data());

  using namespace cuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<256>, 3>>;
  launch_t l;

  auto cache_scores = [=] __device__(int const& tid, int const& bid) {
    cacheCSC_score(tid, pcache[tid], G);
  };
  l.launch_blocked(*scontext, cache_scores, N);
  scontext->synchronize();

  auto score = thrust::reduce(scores.begin(), scores.end());
  return score;
}

template <typename csr_device_t>
__device__ void cacheNbr_score(int v,
                               float& score,
                               csr_device_t& G,
                               int stride = 8) {
  auto num_neighbors = G.get_number_of_neighbors(v);

  if (num_neighbors >= 1) {
    auto start = G.get_starting_edge(v);
    uint32_t unique_sectors = 1;
    // assuming CSR is sorted
    auto prev_sector = G.get_destination_vertex(0) / stride;
    for (auto i = 1; i < num_neighbors; i++) {
      auto cur_sec = G.get_destination_vertex(start + i) / stride;
      if (cur_sec == prev_sector) {
        continue;
      } else {
        unique_sectors++;
      }
    }
    score = (float)unique_sectors / (float)num_neighbors;
  }
}

template <typename csr_device_t>
float avgCacheNbr(
    csr_device_t& G,
    int stride = 8,
    std::shared_ptr<cuda::multi_context_t> context =
        std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0))) {
  std::shared_ptr<cuda::standard_context_t> scontext =
      std::shared_ptr<cuda::standard_context_t>(context->get_context(0));

  int N = G.get_number_of_vertices();
  thrust::device_vector<float> scores(N, 0.0);
  float* pcache = thrust::raw_pointer_cast(scores.data());

  using namespace cuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<256>, 3>>;
  launch_t l;

  auto cache_scores = [=] __device__(int const& tid, int const& bid) {
    cacheNbr_score(tid, pcache[tid], G);
  };
  l.launch_blocked(*scontext, cache_scores, N);
  scontext->synchronize();

  thrust::host_vector<unsigned long long> hpcache(scores);
  for (int i = 0; i < 10; ++i) {
    printf(" %llu ", hpcache[i]);
  }
  
  auto score = thrust::reduce(scores.begin(), scores.end());
  return score / (float)N;
}

template <typename csr_device_t>
__device__ void cacheCSR_score(int i,
                               unsigned long long& score,
                               csr_device_t& G,
                               int stride = 8) {
  auto offsets = G.get_row_offsets();
  auto c2 = offsets[i + 1] / stride;
  auto c1 = offsets[i] / stride;
  score = c2 - c1 + 1;
}

template <typename csr_device_t>
unsigned long long avgCacheLinesCSR(
    csr_device_t& G,
    int stride = 8,
    std::shared_ptr<cuda::multi_context_t> context =
        std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0))) {
  std::shared_ptr<cuda::standard_context_t> scontext =
      std::shared_ptr<cuda::standard_context_t>(context->get_context(0));

  int N = G.get_number_of_vertices();
  thrust::device_vector<unsigned long long> scores(N, 0);
  unsigned long long* pcache = thrust::raw_pointer_cast(scores.data());

  using namespace cuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<256>, 3>>;
  launch_t l;

  auto cache_scores = [=] __device__(int const& tid, int const& bid) {
    cacheCSR_score(tid, pcache[tid], G);
  };
  l.launch_blocked(*scontext, cache_scores, N);
  scontext->synchronize();
  thrust::host_vector<unsigned long long> hpcache(scores);
  for (int i = 0; i < 10; ++i) {
    printf(" %llu ", hpcache[i]);
  }

  auto score = thrust::reduce(scores.begin(), scores.end());
  return score / N;
}

template <typename csc_device_t>
__device__ void aid_score(int v, unsigned long long& score, csc_device_t& G) {
  auto Vnum_neighbors = G.get_number_of_neighbors(v);
  if (Vnum_neighbors >= 2) {
    auto Vstart = G.get_starting_edge(v);
    auto s1 = G.get_source_vertex(Vstart);
    auto s2 = G.get_source_vertex(Vstart + 1);
    score = (s1 > s2) ? s1 - s2 : s2 - s1;

    for (auto Vvi = Vstart + 2; Vvi < Vstart + Vnum_neighbors; Vvi++) {
      auto pVwi = G.get_source_vertex(Vvi - 1);
      auto Vwi = G.get_source_vertex(Vvi);
      score = score + ((pVwi > Vwi) ? pVwi - Vwi : Vwi - pVwi);
    }
    score = score / Vnum_neighbors;
  }
}

template <typename csc_device_t>
unsigned long long aid(
    csc_device_t& G,
    std::shared_ptr<cuda::multi_context_t> context =
        std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0))) {
  std::shared_ptr<cuda::standard_context_t> scontext =
      std::shared_ptr<cuda::standard_context_t>(context->get_context(0));

  int N = G.get_number_of_vertices();
  thrust::device_vector<unsigned long long> scores(N, 0);
  unsigned long long* paid = thrust::raw_pointer_cast(scores.data());

  using namespace cuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<256>, 3>>;
  launch_t l;

  auto aid_scores = [=] __device__(int const& tid, int const& bid) {
    aid_score(tid, paid[tid], G);
  };
  l.launch_blocked(*scontext, aid_scores, N);
  scontext->synchronize();

  auto score = thrust::reduce(scores.begin(), scores.end());
  return score;
}

template <typename csr_device_t>
__device__ void aidCSR_score(int v,
                             unsigned long long& score,
                             csr_device_t& G) {
  auto Vnum_neighbors = G.get_number_of_neighbors(v);
  if (Vnum_neighbors >= 2) {
    auto Vstart = G.get_starting_edge(v);
    auto s1 = G.get_destination_vertex(Vstart);
    auto s2 = G.get_destination_vertex(Vstart + 1);
    score = (s1 > s2) ? s1 - s2 : s2 - s1;

    for (auto Vvi = Vstart + 2; Vvi < Vstart + Vnum_neighbors; Vvi++) {
      auto pVwi = G.get_destination_vertex(Vvi - 1);
      auto Vwi = G.get_destination_vertex(Vvi);
      score = score + ((pVwi > Vwi) ? pVwi - Vwi : Vwi - pVwi);
    }
    score = score / Vnum_neighbors;
  }
}

template <typename csr_device_t>
unsigned long long aidCSR(
    csr_device_t& G,
    std::shared_ptr<cuda::multi_context_t> context =
        std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0))) {
  std::shared_ptr<cuda::standard_context_t> scontext =
      std::shared_ptr<cuda::standard_context_t>(context->get_context(0));

  int N = G.get_number_of_vertices();
  thrust::device_vector<unsigned long long> scores(N, 0);
  unsigned long long* paid = thrust::raw_pointer_cast(scores.data());

  using namespace cuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<256>, 3>>;
  launch_t l;

  auto aid_scores = [=] __device__(int const& tid, int const& bid) {
    aidCSR_score(tid, paid[tid], G);
  };
  l.launch_blocked(*scontext, aid_scores, N);
  scontext->synchronize();

  auto score = thrust::reduce(scores.begin(), scores.end());
  return score;
}

template <typename csr_device_t>
__device__ void uv_gscore(int v, int u, int& score, csr_device_t& G) {
  //  score = 0;
  int N = G.get_number_of_vertices();
  if (u < N) {
    auto Vnum_neighbors = G.get_number_of_neighbors(v);
    auto Vstart = G.get_starting_edge(v);
    for (auto Vvi = Vstart; Vvi < Vstart + Vnum_neighbors; Vvi++) {
      auto Vwi = G.get_destination_vertex(Vvi);
      if (Vwi == u)
        ++score;
    }

    auto Unum_neighbors = G.get_number_of_neighbors(u);
    auto Ustart = G.get_starting_edge(u);
    for (auto Uvi = Ustart; Uvi < Ustart + Unum_neighbors; Uvi++) {
      auto Uwi = G.get_destination_vertex(Uvi);
      if (Uwi == v)
        ++score;
    }
  }
  for (auto h = 0; h < G.get_number_of_vertices(); h++) {
    auto num_neighbors = G.get_number_of_neighbors(h);
    auto start = G.get_starting_edge(h);
    int uv = 0;
    for (auto vi = start; vi < start + num_neighbors; vi++) {
      auto wi = G.get_destination_vertex(vi);
      if (wi == u || wi == v)
        ++uv;
    }
    if (uv == 2)
      score = score + 1;
  }
}

template <typename csr_device_t>
int gscore(
    csr_device_t& G,
    std::shared_ptr<cuda::multi_context_t> context =
        std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0))) {
  std::shared_ptr<cuda::standard_context_t> scontext =
      std::shared_ptr<cuda::standard_context_t>(context->get_context(0));

  int N = G.get_number_of_vertices();
  thrust::device_vector<int> uv_score(N, 0);
  auto uv = thrust::raw_pointer_cast(uv_score.data());

  using namespace cuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<256>, 3>>;
  launch_t l;

  auto adjacent_scores = [=] __device__(int const& tid, int const& bid) {
    auto u = tid;
    auto v = tid + 1;
    uv_gscore(u, v, uv[tid], G);
  };
  l.launch_blocked(*scontext, adjacent_scores, N);
  scontext->synchronize();

  auto score = thrust::reduce(uv_score.begin(), uv_score.end());
  return score;
}

struct is_pad {
  __host__ __device__ bool operator()(const int& x) const { return x == -1; }
};

template <typename coo_device_t>
void apply_permutation(coo_device_t& G,
                       coo_device_t& rG,
                       std::shared_ptr<cuda::multi_context_t> context,
                       thrust::device_vector<int>& perm) {
  std::shared_ptr<cuda::standard_context_t> scontext =
      std::shared_ptr<cuda::standard_context_t>(context->get_context(0));

  int M = G.number_of_nonzeros;
  int N = G.number_of_rows;

  auto I = thrust::raw_pointer_cast(G.row_indices.data());
  auto J = thrust::raw_pointer_cast(G.column_indices.data());

  auto rI = thrust::raw_pointer_cast(rG.row_indices.data());
  auto rJ = thrust::raw_pointer_cast(rG.column_indices.data());

  auto permP = thrust::raw_pointer_cast(perm.data());

  thrust::device_vector<int> inverse(N);
  auto ip = thrust::raw_pointer_cast(inverse.data());

  auto invert = [=] __device__(int const& tid, int const& bid) {
    ip[permP[tid]] = tid;
  };
  auto permute = [=] __device__(int const& tid, int const& bi) {
    rI[tid] = ip[I[tid]];
    rJ[tid] = ip[J[tid]];
  };

  using namespace cuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<256>, 3>>;
  launch_t l;

  l.launch_blocked(*scontext, invert, (std::size_t)N);
  scontext->synchronize();

  l.launch_blocked(*scontext, permute, (std::size_t)M);
  scontext->synchronize();
}

template <typename coo_device_t, typename coo_host_t>
void edge_order(coo_device_t& G,
                coo_device_t& rG,
                coo_host_t& Gh,
                std::shared_ptr<cuda::multi_context_t> context) {
  int N = Gh.number_of_rows;
  int M = Gh.number_of_nonzeros;
  auto I = thrust::raw_pointer_cast(Gh.row_indices.data());
  auto J = thrust::raw_pointer_cast(Gh.column_indices.data());

  //  printf("%i %i\n",Gh.row_indices.size(),Gh.row_indices.size());
  thrust::host_vector<int> Seen(N, 0);
  thrust::host_vector<int> perm(N);

  /*  for(int i = 0, j = 0, k = N-1; i < M; ++i) {
    auto a = *(I+i);
    auto b = *(J+i);
    if(!Seen[a] && !Seen[b]) {
      Seen[a] = 1;
      perm[j++] = a;
      Seen[b] = 1;
      perm[j++] = b;
    }
    else if(!Seen[a]){
      Seen[a] = 1;
      perm[k--] = a;
    }
    else if(!Seen[b]){
      Seen[b] = 1;
      perm[k--] = b;
    }
    }*/
  int j = 0;
  int ii = 0;
  for (int i = 0; i < M; ++i) {
    auto a = I[i];
    auto b = J[i];
    if (a != b) {
      // if(a >= N || b >= N) printf("A %i B %i \n",a,b);
      if (!Seen[a] && !Seen[b]) {
        Seen[a] = 1;
        ++ii;
        perm[j++] = a;
        if (a != b) {
          Seen[b] = 1;
          perm[j++] = b;
        }
      }
    }
  }

  //  fflush(stdout);

  for (int i = 0; i < M; ++i) {
    auto a = I[i];
    auto b = J[i];
    if (a != b) {
      if (!Seen[a]) {
        ++ii;
        Seen[a] = 1;
        perm[j++] = a;
      } else if (!Seen[b]) {
        Seen[b] = 1;
        perm[j++] = b;
      }
    }
  }

  thrust::device_vector<int> permutation(N);
  permutation = perm;
  printf("N = %i, M = %i, VC found = %i, |I| = %i \n", N, M, j, ii);
  apply_permutation(G, rG, context, permutation);
}

template <typename coo_device_t>
void uniquify2(coo_device_t& G,
               coo_device_t& rG,
               std::shared_ptr<cuda::multi_context_t> context) {
  int N = G.number_of_rows;
  int M = G.number_of_nonzeros;

  // test
  thrust::device_vector<int> rperm(M);
  thrust::default_random_engine g(3);

  thrust::sequence(rperm.begin(), rperm.end());
  thrust::shuffle(rperm.begin(), rperm.end(), g);
  thrust::sort_by_key(rperm.begin(), rperm.end(), G.row_indices.begin());
  thrust::sort_by_key(rperm.begin(), rperm.end(), G.column_indices.begin());
  thrust::sort_by_key(rperm.begin(), rperm.end(), rG.row_indices.begin());
  thrust::sort_by_key(rperm.begin(), rperm.end(), rG.column_indices.begin());

  // test

  thrust::device_vector<int> permutation(2 * M);
  thrust::device_vector<int> IJ(2 * M);
  thrust::copy(G.row_indices.begin(), G.row_indices.end(), IJ.begin());
  thrust::copy(G.column_indices.begin(), G.column_indices.end(),
               IJ.begin() + M);

  thrust::sequence(permutation.begin(), permutation.end());

  thrust::sort_by_key(IJ.begin(), IJ.end(), permutation.begin());

  auto nend = thrust::unique_by_key(thrust::device, IJ.begin(), IJ.end(),
                                    permutation.begin());

  thrust::sort_by_key(permutation.begin(), permutation.begin() + N, IJ.begin());

  apply_permutation(G, rG, context, IJ);
  //  thrust::gather(IJ.begin(), IJ.begin() + N, G.row_indices.begin(),
  //             rG.row_indices.begin());
  // thrust::gather(IJ.begin(), IJ.begin() + N, G.column_indices.begin(),
  //             rG.column_indices.begin());
  /*  thrust::sort(IJ.begin(), IJ.end());
  printf("NN = %i Reduce = %i
  \n",999*998/2,thrust::reduce(IJ.begin(),IJ.begin()+1000));
  gunrock::print::head(IJ, 40, "IJ-permvector");
  */
}

template <typename coo_device_t>
void degree(coo_device_t& G,
            coo_device_t& rG,
            std::shared_ptr<cuda::multi_context_t> context) {
  int M = G.number_of_nonzeros;
  int N = G.number_of_rows;
  thrust::device_vector<int> ones(2 * M, 1);
  //  thrust::device_vector<int> degR(M, 0);
  thrust::device_vector<int> outKey(N);
  // thrust::device_vector<int> degC(M, 0);
  thrust::device_vector<int> deg(N);
  thrust::device_vector<int> permutation(N);
  thrust::sequence(permutation.begin(), permutation.end());

  thrust::device_vector<int> IJ(2 * M);
  thrust::copy(G.row_indices.begin(), G.row_indices.end(), IJ.begin());
  thrust::copy(G.column_indices.begin(), G.column_indices.end(),
               IJ.begin() + M);

  thrust::sort(IJ.begin(), IJ.end());
  auto endn = thrust::reduce_by_key(thrust::device, IJ.begin(), IJ.end(),
                                    ones.begin(), outKey.begin(), deg.begin());

  //  thrust::transform(degR.begin(), degR.end(), degC.begin(), deg.begin(),
  //                thrust::plus<int>());

  thrust::sort_by_key(deg.begin(), deg.begin(), permutation.begin(),
                      thrust::greater<int>());

  apply_permutation(G, rG, context, permutation);
}

template <typename coo_device_t>
void random(coo_device_t& G,
            coo_device_t& rG,
            std::shared_ptr<cuda::multi_context_t> context) {
  int M = G.number_of_nonzeros;
  // test
  /*
  thrust::device_vector<int> rperm(M);
  thrust::default_random_engine gg(1);

  thrust::sequence(rperm.begin(), rperm.end());
  thrust::shuffle(rperm.begin(), rperm.end(), gg);
    thrust::device_vector<int> rperm1(rperm);
    thrust::device_vector<int> rperm2(rperm);
    thrust::device_vector<int> rperm3(rperm);
  thrust::sort_by_key(rperm.begin(), rperm.end(), G.row_indices.begin());
  thrust::sort_by_key(rperm1.begin(), rperm1.end(), G.column_indices.begin());
  thrust::sort_by_key(rperm2.begin(), rperm2.end(), rG.row_indices.begin());
  thrust::sort_by_key(rperm3.begin(), rperm3.end(), rG.column_indices.begin());
  //test
  */
  int N = G.number_of_rows;
  thrust::device_vector<int> permutation(N);
  thrust::default_random_engine g(2);

  thrust::sequence(permutation.begin(), permutation.end());
  thrust::shuffle(permutation.begin(), permutation.end(), g);

  //  thrust::sort(permutation.begin(), permutation.end());
  // printf("PERM SUM %i\n",
  // thrust::reduce(permutation.begin(),permutation.end()));
  // thrust::gather(permutation.begin(), permutation.end(),
  // G.row_indices.begin(),
  //             rG.row_indices.begin());
  // thrust::gather(permutation.begin(), permutation.end(),
  //             G.column_indices.begin(), rG.column_indices.begin());
  apply_permutation(G, rG, context, permutation);
}

template <typename coo_device_t>
void uniquify(coo_device_t& G,
              coo_device_t& rG,
              std::shared_ptr<cuda::multi_context_t> context) {
  //
  std::shared_ptr<cuda::standard_context_t> scontext =
      std::shared_ptr<cuda::standard_context_t>(context->get_context(0));
  int M = G.number_of_nonzeros;
  int N = G.number_of_rows;
  int MM = 2 * M;

  // test

  thrust::device_vector<int> rperm1(G.column_indices), rperm2(G.column_indices),
      rperm3(G.column_indices), rperm4(G.column_indices);
  //  thrust::default_random_engine g(1);
  // rperm = G.column_indices;
  // thrust::sequence(rperm.begin(), rperm.end());
  // thrust::shuffle(rperm.begin(), rperm.end(), g);
  /*
  thrust::sort_by_key(rperm1.begin(), rperm1.end(), G.row_indices.begin());
  thrust::sort_by_key(rperm2.begin(), rperm2.end(), G.column_indices.begin());
  thrust::sort_by_key(rperm3.begin(), rperm3.end(), rG.row_indices.begin());
  thrust::sort_by_key(rperm4.begin(), rperm4.end(), rG.column_indices.begin());
  */
  // test

  thrust::device_vector<int> dkeys(N, std::numeric_limits<int>::max());
  thrust::device_vector<int> zp(MM, -1);

  auto I = thrust::raw_pointer_cast(G.row_indices.data());
  auto J = thrust::raw_pointer_cast(G.column_indices.data());

  auto rI = thrust::raw_pointer_cast(rG.row_indices.data());
  auto rJ = thrust::raw_pointer_cast(rG.column_indices.data());

  int* pk = thrust::raw_pointer_cast(dkeys.data());
  int* p = thrust::raw_pointer_cast(zp.data());

  auto make_keys = [=] __device__(int const& tid, int const& bid) {
    if (tid < M) {
      // pk[I[tid]] = pk[I[tid]] > tid ? tid : pk[I[tid]];
      // if (pk[I[tid]] == std::numeric_limits<int>::max())
      // pk[I[tid]] = tid;
      // if(pk[I[tid]] > tid)
      // pk[I[tid]] = tid;
      atomicMin(&(pk[I[tid]]), tid);
    } else {
      // pk[J[tid - M]] = pk[J[tid - M]] > tid ? tid : pk[J[tid - M]];
      // if (pk[J[tid - M]] == std::numeric_limits<int>::max())
      // if(pk[J[tid-M]] > tid)
      //	pk[J[tid - M]] = tid;
      atomicMin(&(pk[J[tid - M]]), tid);
    }
  };

  auto make_zperm = [=] __device__(int const& tid, int const& bid) {
    if (pk[tid] != std::numeric_limits<int>::max())
      p[pk[tid]] = tid;
  };

  using namespace cuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<1024>, 3>>;
  launch_t l;

  l.launch_blocked(*scontext, make_keys, (std::size_t)MM);
  scontext->synchronize();

  l.launch_blocked(*scontext, make_zperm, (std::size_t)N);
  scontext->synchronize();

  // zp.erase(thrust::remove_if(zp.begin(), zp.end(), is_pad()), zp.end());
  thrust::remove_if(zp.begin(), zp.end(), is_pad());
  auto z = thrust::raw_pointer_cast(zp.data());

  thrust::device_vector<int> iz(N);
  auto izp = thrust::raw_pointer_cast(iz.data());

  auto inverse = [=] __device__(int const& tid, int const& bid) {
    izp[z[tid]] = tid;
  };
  auto permute = [=] __device__(int const& tid, int const& bi) {
    rI[tid] = izp[I[tid]];
    rJ[tid] = izp[J[tid]];
  };

  l.launch_blocked(*scontext, inverse, (std::size_t)N);
  scontext->synchronize();

  l.launch_blocked(*scontext, permute, (std::size_t)M);
  scontext->synchronize();

  /*
  thrust::device_vector<int> rperm(M);
  //thrust::default_random_engine g(3);

  thrust::sequence(rperm.begin(), rperm.end());
  //thrust::shuffle(rperm.begin(), rperm.end(), g);
  thrust::sort_by_key(rG.row_indices.begin(), rG.row_indices.end(),
  rperm.begin()); thrust::sort_by_key(rperm.begin(), rperm.end(),
  rG.row_indices.begin()); thrust::sort_by_key(rperm.begin(), rperm.end(),
  rG.column_indices.begin());
  */

  /*thrust::sort(zp.begin(),zp.end());
  printf("NN = %i Reduce = %i
  \n",999*1000/2,thrust::reduce(zp.begin(),zp.begin()+1000));
  gunrock::print::head(zp, 40, "IJ-permvector");*/
}
template <typename coo_device_t>
void uniquify_strided(coo_device_t& G,
                      coo_device_t& rG,
                      std::shared_ptr<cuda::multi_context_t> context) {
  //
  std::shared_ptr<cuda::standard_context_t> scontext =
      std::shared_ptr<cuda::standard_context_t>(context->get_context(0));
  int M = G.number_of_nonzeros;
  int N = G.number_of_rows;
  int MM = 2 * M;

  // test

  // thrust::device_vector<int> rperm1(G.column_indices),
  // rperm2(G.column_indices),rperm3(G.column_indices),rperm4(G.column_indices);

  thrust::device_vector<int> dkeys(N, std::numeric_limits<int>::max());
  thrust::device_vector<int> zp(MM, -1);

  auto I = thrust::raw_pointer_cast(G.row_indices.data());
  auto J = thrust::raw_pointer_cast(G.column_indices.data());

  auto rI = thrust::raw_pointer_cast(rG.row_indices.data());
  auto rJ = thrust::raw_pointer_cast(rG.column_indices.data());

  int* pk = thrust::raw_pointer_cast(dkeys.data());
  int* p = thrust::raw_pointer_cast(zp.data());

  auto make_keys = [=] __device__(int const& tid, int const& bid) {
    if (tid < M) {
      if (pk[I[tid]] > tid)
        pk[I[tid]] = tid;
    } else {
      if (pk[J[tid - M]] > tid)
        pk[J[tid - M]] = tid;
    }
  };

  auto make_zperm = [=] __device__(int const& tid, int const& bid) {
    if (pk[tid] != std::numeric_limits<int>::max())
      p[pk[tid]] = tid;
  };

  using namespace cuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<256>, 3>>;
  launch_t l;

  l.launch_blocked(*scontext, make_keys, (std::size_t)MM);
  scontext->synchronize();

  l.launch_blocked(*scontext, make_zperm, (std::size_t)N);
  scontext->synchronize();

  // zp.erase(thrust::remove_if(zp.begin(), zp.end(), is_pad()), zp.end());
  thrust::remove_if(zp.begin(), zp.end(), is_pad());
  auto z = thrust::raw_pointer_cast(zp.data());

  thrust::device_vector<int> iz(N);
  auto izp = thrust::raw_pointer_cast(iz.data());

  thrust::device_vector<int> trans(N);
  auto transP = thrust::raw_pointer_cast(trans.data());

  auto inverse = [=] __device__(int const& tid, int const& bid) {
    izp[z[tid]] = tid;
  };
  auto permute = [=] __device__(int const& tid, int const& bi) {
    //  rI[tid] = izp[I[tid]];
    // rJ[tid] = izp[J[tid]];
    rI[tid] = transP[I[tid]];
    rJ[tid] = transP[J[tid]];
  };

  auto transpose = [=] __device__(int const& tid, int const& bi) {
    if (tid % 8 == 0) {
      transP[tid] = izp[(tid + 8) % N];
      // transP[(tid + 8) % N] = izp[tid];
    } else
      transP[tid] = izp[tid];
  };

  l.launch_blocked(*scontext, inverse, (std::size_t)N);
  scontext->synchronize();

  l.launch_blocked(*scontext, transpose, (std::size_t)N);
  scontext->synchronize();

  l.launch_blocked(*scontext, permute, (std::size_t)M);
  scontext->synchronize();
}
}  // namespace reorder
}  // namespace graph
}  // namespace gunrock
