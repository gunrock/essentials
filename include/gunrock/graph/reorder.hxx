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

template <typename csr_device_t>
__device__ void uv_gscore(int v, int u, int& score, csr_device_t& G) {
  //  score = 0;
  int N = G.get_number_of_vertices();
  if(u < N){
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
int gscore(csr_device_t& G, std::shared_ptr<cuda::multi_context_t> context = std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0))) {
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

template <typename coo_device_t>
void uniquify2(coo_device_t& G,
               coo_device_t& rG,
               std::shared_ptr<cuda::multi_context_t> context) {
  int N = G.number_of_rows;
  int M = G.number_of_nonzeros;
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
                      thrust::less<int>());

  apply_permutation(G, rG, context, permutation);
}

template <typename coo_device_t>
void random(coo_device_t& G,
            coo_device_t& rG,
            std::shared_ptr<cuda::multi_context_t> context) {
  int N = G.number_of_rows;
  thrust::device_vector<int> permutation(N);
  thrust::default_random_engine g(2);

  thrust::sequence(permutation.begin(), permutation.end());
  thrust::shuffle(permutation.begin(), permutation.end(), g);

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

  thrust::device_vector<int> dkeys(N, std::numeric_limits<int>::max());
  thrust::device_vector<int> zp(MM, -1);

  auto I = thrust::raw_pointer_cast(G.row_indices.data());
  auto J = thrust::raw_pointer_cast(G.column_indices.data());

  auto rI = thrust::raw_pointer_cast(rG.row_indices.data());
  auto rJ = thrust::raw_pointer_cast(rG.column_indices.data());

  int* pk = thrust::raw_pointer_cast(dkeys.data());
  int* p = thrust::raw_pointer_cast(zp.data());

  auto make_keys = [=] __device__(int const& tid, int const& bid) {
    if (tid < M)
      pk[I[tid]] = pk[I[tid]] > tid ? tid : pk[I[tid]];
    else {
      pk[J[tid - M]] = pk[J[tid - M]] > tid ? tid : pk[J[tid - M]];
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
  /*thrust::sort(zp.begin(),zp.end());
  printf("NN = %i Reduce = %i
  \n",999*1000/2,thrust::reduce(zp.begin(),zp.begin()+1000));
  gunrock::print::head(zp, 40, "IJ-permvector");*/
}
}  // namespace reorder
}  // namespace graph
}  // namespace gunrock
