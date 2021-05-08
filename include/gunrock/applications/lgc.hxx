/**
 * @file lgc.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Original implementation by: Ben Johnson (bkj.322@gmail.com)
 * @version 0.1
 * @date 2021-05-03
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <gunrock/applications/application.hxx>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

namespace gunrock {
namespace lgc {

template <typename vertex_t, typename weight_t>
struct param_t {
  graph::vertex_pair_t<vertex_t> pair;

  weight_t eps;    // Tolerance for convergence
  weight_t alpha;  // Parameterizes conductance/size of output cluster
  weight_t rho;    // Parameterizes conductance/size of output cluster

  param_t(vertex_t _source,
          vertex_t _source_neighbor,
          weight_t _eps,
          weight_t _alpha,
          weight_t _rho)
      : eps(_eps), alpha(_alpha), rho(_rho) {
    pair.source = _source;
    pair.destination = _source_neighbor;
  }
};

template <typename weight_t>
struct result_t {
  weight_t* p;  // Truncated z-values && also the output.
  result_t(weight_t* _p) : p(_p) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<weight_t> gradient;  // Gradient values
  thrust::device_vector<weight_t> y;         // Intermediate quantity
  thrust::device_vector<weight_t> z;         // Intermediate quantity
  thrust::device_vector<weight_t> q;         // Truncated z-values
  thrust::device_vector<int> visited;        // track

  int num_ref_nodes = 1;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<cuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    gradient.resize(n_vertices);
    y.resize(n_vertices);
    z.resize(n_vertices);
    q.resize(n_vertices);
    visited.resize(n_vertices);
  }

  void reset() override {
    // Execution policy for a given context (using single-gpu).
    auto policy = this->context->get_context(0)->execution_policy();
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    thrust::fill_n(policy, gradient.begin(), n_vertices, weight_t(0.0f));
    thrust::fill_n(policy, y.begin(), n_vertices, weight_t(0.0f));
    thrust::fill_n(policy, z.begin(), n_vertices, weight_t(0.0f));
    thrust::fill_n(policy, q.begin(), n_vertices, weight_t(0.0f));
    thrust::fill_n(policy, visited.begin(), n_vertices, 0);
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  using gunrock::enactor_t<problem_t>::enactor_t;

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void prepare_frontier(frontier_t<vertex_t>* f,
                        cuda::multi_context_t& context) override {
    auto P = this->get_problem();
    f->push_back(P->param.pair.source);
    f->push_back(P->param.pair.destination);
  }

  void loop(cuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto f = E->get_input_frontier();

    auto n_vertices = G.get_number_of_vertices();

    auto gradient = P->gradient.data().get();
    auto y = P->y.data().get();
    auto z = P->z.data().get();
    auto q = P->q.data().get();

    auto visited = P->visited.data().get();

    auto alpha = P->param.alpha;
    auto rho = P->param.rho;

    auto pair = P->param.pair;

    auto iteration = this->iteration;

    auto policy = this->context->get_context(0)->execution_policy();

    auto num_ref_nodes = P->num_ref_nodes;

    // compute operation
    auto compute_op = [=] __host__ __device__(vertex_t v) {
      printf("v = %i\n", v);

      if (!gunrock::util::limits::is_valid(v))
        return gunrock::numeric_limits<vertex_t>::invalid();

      // ignore the neighbor on the first iteration
      if ((iteration == 0) && (v == pair.destination))
        return gunrock::numeric_limits<vertex_t>::invalid();

      // Compute degrees
      auto degree = G.get_number_of_neighbors(v);
      weight_t degree_sqrt = sqrt((weight_t)degree);
      weight_t inv_degree_sqrt = 1.0 / degree_sqrt;

      printf("degree_sqrt, inv_degree_sqrt, degree = %lf, %lf, %i\n",
             degree_sqrt, inv_degree_sqrt, degree);

      // this is at end in original implementation, but works
      // here after the first iteration (+ have to adjust for
      // it in StopCondition)
      if ((iteration > 0) && (v == pair.source)) {
        gradient[v] -= alpha / num_ref_nodes * inv_degree_sqrt;
      }

      z[v] = y[v] - gradient[v];
      printf("gradient[v] = %lf\n", gradient[v]);

      if (z[v] == 0)
        return gunrock::numeric_limits<vertex_t>::invalid();

      weight_t q_old = q[v];
      weight_t thresh = rho * alpha * degree_sqrt;

      if (z[v] >= thresh) {
        q[v] = z[v] - thresh;
      } else if (z[v] <= -thresh) {
        q[v] = z[v] + thresh;
      } else {
        q[v] = (weight_t)0.0f;
      }

      printf("q[v] = %lf\n", q[v]);

      if (iteration == 0) {
        y[v] = q[v];
      } else {
        weight_t beta = (1 - sqrt(alpha)) / (1 + sqrt(alpha));
        y[v] = q[v] + beta * (q[v] - q_old);
      }

      visited[v] = 0;
      gradient[v] = y[v] * (1.0 + alpha) / 2;
      return gunrock::numeric_limits<vertex_t>::invalid();
    };

    thrust::transform(policy, f->begin(), f->end(),
                      thrust::make_discard_iterator(), compute_op);

    auto spread_op = [=] __host__ __device__(
                         vertex_t const& src, vertex_t const& dst,
                         edge_t const& edge, weight_t const& weight) -> bool {
      weight_t src_dn_sqrt =
          1.0 / sqrt((weight_t)G.get_number_of_neighbors(src));
      weight_t dest_dn_sqrt =
          1.0 / sqrt((weight_t)G.get_number_of_neighbors(dst));
      weight_t src_y = y[src];

      weight_t grad_update =
          -src_dn_sqrt * src_y * dest_dn_sqrt * (1.0 - alpha) / 2;
      weight_t last_grad = math::atomic::add(gradient + dst, grad_update);
      if (last_grad + grad_update == 0)
        return false;

      bool already_touched = math::atomic::max(visited + dst, 1) == 1;
      return !already_touched;
    };

    operators::advance::execute<operators::load_balance_t::merge_path>(
        G, E, spread_op, context);
  }

  virtual bool is_converged(cuda::multi_context_t& context) {
    // never break on first iteration
    if (this->iteration == 0)
      return false;

    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto f = E->get_input_frontier();

    auto n_vertices = G.get_number_of_vertices();
    auto num_ref_nodes = P->num_ref_nodes;
    auto gradient = P->gradient.data().get();

    auto alpha = P->param.alpha;
    auto rho = P->param.rho;
    auto eps = P->param.eps;
    auto pair = P->param.pair;

    auto q = P->q.data().get();
    auto p = P->result.p;

    thrust::device_vector<int> grad_scale(1, 0);
    thrust::device_vector<weight_t> grad_scale_value(1, (weight_t)0.0f);

    auto d_grad_scale = grad_scale.data().get();
    auto d_grad_scale_value = grad_scale_value.data().get();

    weight_t grad_thresh = rho * alpha * (1 + eps);

    auto convergence_op = [=] __host__ __device__(vertex_t const& v) {
      weight_t v_dn_sqrt = 1.0f / sqrt((weight_t)G.get_number_of_neighbors(v));
      weight_t val = gradient[v];

      if (v == pair.source)
        val -= (alpha / num_ref_nodes) * v_dn_sqrt;

      val = abs(val * v_dn_sqrt);

      math::atomic::max(d_grad_scale_value, val);
      if (val > grad_thresh) {
        math::atomic::max(d_grad_scale, 1);
      }

      return 0;  // ignored.
    };

    auto policy = this->context->get_context(0)->execution_policy();
    thrust::transform(policy, f->begin(), f->end(),
                      thrust::make_discard_iterator(), convergence_op);

    thrust::host_vector<int> check_grad_scale = grad_scale;
    // thrust::host_vector<weight_t> check_grad_scale_value = grad_scale_value;

    // gradient too small:: converged.
    if (!(check_grad_scale[0]) && this->iteration == 100) {
      auto scale_op = [=] __device__(vertex_t const& v) -> void {
        p[v] = abs(q[v] * sqrt((weight_t)G.get_number_of_neighbors(v)));
        printf("%lf, %lf, %lf\n", p[v], q[v],
               sqrt((weight_t)G.get_number_of_neighbors(v)));
      };

      operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
          G,         // graph
          scale_op,  // lambda function
          context    // context
      );
      return true;
    }

    return false;
  }

};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type source,
          typename graph_t::vertex_type source_neighbor,
          typename graph_t::weight_type eps,
          typename graph_t::weight_type phi,
          typename graph_t::weight_type vol,
          typename graph_t::weight_type* p  // Output
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t, weight_t>;
  using result_type = result_t<weight_t>;

  // Calculate alpha
  weight_t num_edges = G.get_number_of_edges();
  weight_t log_num_edges = log2(num_edges);
  weight_t alpha = pow(phi, 2) / (255.0f * log(100.0f * sqrt(num_edges)));

  // Calculate rho
  weight_t rho = 1 / (pow(2, min(1 + log2(vol), log_num_edges)));
  rho *= (1.0 / (48.0 * log_num_edges));

  param_type param(source, source_neighbor, eps, alpha, rho);
  result_type result(p);
  // </user-defined>

  // <boiler-plate>
  auto multi_context =
      std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0));

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, multi_context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, multi_context);
  return enactor.enact();
  // </boiler-plate>
}

}  // namespace lgc
}  // namespace gunrock