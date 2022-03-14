/**
 * @file thread_mapped.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Advance operator where a vertex/edge is mapped to a thread.
 * @version 0.1
 * @date 2020-10-20
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/util/math.hxx>
#include <gunrock/cuda/cuda.hxx>

#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/framework/operators/for/for.hxx>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

namespace gunrock {
namespace operators {
namespace advance {
namespace variance_sort {

struct and_func : public thrust::unary_function<int, int> {
  __host__ __device__ int operator()(std::size_t i) { return 8 & x; }
};

template <advance_direction_t direction,
          advance_io_type_t input_type,
          advance_io_type_t output_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t& input,
             frontier_t& output,
             work_tiles_t& segments,
             cuda::standard_context_t& context) {
  using type_t = typename frontier_t::type_t;
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;

  error::throw_if_exception(input_type == advance_io_type_t::graph,
                            "Input type is not supported.");

  std::size_t num_elements = input.get_number_of_elements();

  auto variance = [=] __device__(std::size_t i) {
    auto v = input.get_element_at(i);
    // if item is invalid, segment size is 0.
    if (!gunrock::util::limits::is_valid(v))
      return edge_t(0);
    else
      return G.get_number_of_neighbors(v);
  };

  thrust::sort_by_key(
      context.execution_policy(),
      thrust::make_transform_iterator(
          thrust::make_counting_iterator<std::size_t>(0), variance),
      thrust::make_transform_iterator(
          thrust::make_counting_iterator<std::size_t>(num_elements), variance),
      input.begin());
  context.synchronize();

  if (output_type != advance_io_type_t::none) {
    auto size_of_output = compute_output_offsets(G, &input, segments, context);

    // If output frontier is empty, resize and return.
    if (size_of_output <= 0) {
      output.set_number_of_elements(0);
      return;
    }

    /// Resize the output (inactive) buffer to the new size.
    /// @todo Can be hidden within the frontier struct.
    if (output.get_capacity() < size_of_output)
      output.reserve(size_of_output);
    output.set_number_of_elements(size_of_output);
  }

  // Get output data of the active buffer.
  auto segments_ptr = segments.data().get();

  auto thread_mapped = [=] __device__(int const& tid, int const& bid) {
    auto v = input.get_element_at(tid);

    if (!gunrock::util::limits::is_valid(v))
      return;

    auto starting_edge = G.get_starting_edge(v);
    auto total_edges = G.get_number_of_neighbors(v);

    for (auto i = 0; i < total_edges; ++i) {
      auto e = i + starting_edge;            // edge id
      auto n = G.get_destination_vertex(e);  // neighbor id
      auto w = G.get_edge_weight(e);         // weight
      bool cond = op(v, n, e, w);

      if (output_type != advance_io_type_t::none) {
        std::size_t out_idx = segments_ptr[tid] + i;
        type_t element =
            (cond && n != v) ? n : gunrock::numeric_limits<type_t>::invalid();
        output.set_element_at(element, out_idx);
      }
    }
  };

  // Set-up and launch thread-mapped advance.
  using namespace cuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<256>, 3>>;

  launch_t l;
  l.launch_blocked(context, thread_mapped, num_elements);
  context.synchronize();
}
}  // namespace variance_sort
}  // namespace advance
}  // namespace operators
}  // namespace gunrock
