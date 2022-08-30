#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace compact {

template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             gcuda::standard_context_t& context) {
  using vertex_t = typename graph_t::vertex_type;
  using size_type = decltype(input->get_number_of_elements());

  assert(false);
  return;
}
}  // namespace compact
}  // namespace filter
}  // namespace operators
}  // namespace gunrock
