# Copyright (C) 2022, Thibaud Ehret <ehret.thibaud@gmail.com>

import scipy
import jax.numpy as jnp
import jax.scipy as jscipy
import jax
import jax.experimental.sparse as sparse
from functools import partial

from jax.experimental.host_callback import id_print

# Compute the determinant of the matrix represented by its Cholesky decomposition (vectorized)
# chol: Cholesky decomposition  of a given matrix, (dim, dim) float32
@partial(jax.vmap, in_axes=0)
def half_log_det_chol(chol):
    return jnp.sum(jnp.log(jnp.diagonal(chol)))

# Compute the probability of x in the GMM represented by means and precisions_chol (vectorized)
# x: data point (dim,) float32
# means: means of the GMM, (size_mixture, dim) float32
# inv_covs_chol: Cholesky decomposition of the inverse of the covariance, (size_mixture, dim, dim) float32
@partial(jax.vmap, in_axes=(0, None, None))
def log_pdf(x, means, inv_covs_chol):
    # We need the det of the cov. matrix, not the det of the inv. cov.
    # but we use the -0.5 coefficient in the return to normalize
    log_det = half_log_det_chol(inv_covs_chol)

    def log_gaussian(x, mu, icc):
        y = jnp.dot(icc, x - mu)
        return jnp.sum(y**2)
    log_prob = jax.vmap(log_gaussian, in_axes=(None, 0, 0))(x, means, inv_covs_chol)

    # +log_det instead of -0.5log_det because we computed the 1/2 det of the Cholesky of the inverse matrix
    return -0.5 * (means.shape[1] * jnp.log(2 * jnp.pi) + log_prob) + log_det

@partial(jax.vmap, in_axes=(None, 1))
def estimate_gaussian(X, w):
    w = jnp.exp(w)
    # TODO robustify better than clip (mechanism to drop gaussians)
    norm = jnp.sum(w).clip(min=1)
    mean = jnp.sum(w[:,None] * X, axis=0) / norm
    diff = X - mean[None,:]
    # TODO regularization step
    cov = jnp.dot(diff.T, w[:,None] * diff) / norm + 1e-6*jnp.eye(X.shape[1])
    return mean, cov

# Return:
#   means:
#   covs:
#   weights:
#   converged:
@partial(jax.jit, static_argnums=(4,5))
def gmm_fit(X, mean_init, covs_init, weights_init, max_iter=100, tol=1e-3):
    def cond_fun(data):
        (_, _, _, prev_lower_bound, curr_lower_bound, it) = data
        return jnp.logical_and(it < max_iter, jnp.abs(curr_lower_bound - prev_lower_bound) > tol)

    # OPTI better to give X as a parameter or keep it like this?
    def body_fun(data):
        (means, covs, weights, prev_lower_bound, curr_lower_bound, it) = data
        prev_lower_bound = curr_lower_bound

        dim = covs.shape[1]
        covs_chol = jnp.linalg.cholesky(covs)
        # OPTI check if more efficient to do that or solve X-mu like Aitor did
        # Output of jnp.linalg.cholesky is lower triangular
        inv_covs_chol = jax.vmap(lambda c,e: jscipy.linalg.solve_triangular(c,e,lower=True), in_axes=(0,None))(covs_chol, jnp.eye(dim))

        log_weighted_probas = log_pdf(X, means, inv_covs_chol) + jnp.log(weights[None,:])
        norm = jscipy.special.logsumexp(log_weighted_probas, axis=1)
        curr_lower_bound = jnp.mean(norm)
        log_weighted_probas = log_weighted_probas - norm[:,None]

        means, covs = estimate_gaussian(X, log_weighted_probas)
        weights = jscipy.special.logsumexp(log_weighted_probas, axis=0)
        weights = jnp.exp(weights)
        weights = weights / jnp.sum(weights, keepdims=True)
        return (means, covs, weights, prev_lower_bound, curr_lower_bound, it+1)

    means, covs, weights, prev_lower_bound, curr_lower_bound, it = jax.lax.while_loop(cond_fun, body_fun, (mean_init, covs_init, weights_init, -jnp.inf, jnp.inf, 0))
    return means, covs, weights, jnp.abs(prev_lower_bound - curr_lower_bound) < tol

# initialize the GMM using K-means
# INPUT
# rng: rng state
# X: (N, dim) numpy style float32 array
# n: int
# n_try: int
# tol: float32
# RETURN
# means: 
# covs: 
# weights: 
@partial(jax.jit, static_argnums=(2,3,4,5))
def kmeans_init(key, X, n, n_try=1, max_iter=100, tol=1e-3):
    @partial(jax.vmap, in_axes=(0, None))
    def assign(x, means):
        idx = jnp.argmin(jnp.linalg.norm(means - x[None,:], axis=-1))
        dist = jnp.linalg.norm(means[idx,:] - x)
        return idx, dist

    def kmeans(key, points, tol=1e-6):
        def cond_fun(data):
            (_, curr_dist, prev_dist, it) = data
            return jnp.logical_and(it < max_iter, jnp.abs(curr_dist - prev_dist) > tol)

        def body_fun(data):
            means, prev_dist, _, it = data
            idxs, dists = assign(points, means)

            #counts = jax.ops.segment_sum(jnp.ones((points.shape[0],)), idxs, num_segments=means.shape[0])
            # Found the version below by Jean-Baptiste Cordonnier that might be faster
            # Adapted from https://colab.research.google.com/drive/1AwS4haUx6swF82w3nXr6QKhajdF8aSvA#scrollTo=LJyoi46rIJr7
            # OPTI check if really faster than the one above
            counts = (idxs[None, :] == jnp.arange(n)[:, None]).sum(axis=1, keepdims=True).clip(min=1.)

            #means = jax.ops.segment_sum(points, idxs, num_segments=means.shape[0])/counts
            # Same as above
            means = jnp.sum(
                jnp.where(
                    idxs[:, None, None] == jnp.arange(n)[None, :, None],
                    points[:, None, :],
                    0.,
                ), axis=0) / counts

            return (means, jnp.mean(dists), prev_dist, it+1)

        # Draw k samples at random from all samples to initialize
        idxs_init = jax.random.permutation(key, points.shape[0])[:n]
        means, dist, _, _ = jax.lax.while_loop(cond_fun, body_fun, (points[idxs_init, :], jnp.inf, -jnp.inf, 0))
        return means, dist 

    # Do n_try (different) estimations, keep the best
    init_means, init_dist = jax.vmap(lambda key: kmeans(key, X, tol))(jax.random.split(key, n_try))
    i = jnp.argmin(init_dist)

    means = init_means[i]
    idxs, _ = assign(X, means)

    # OPTI cf previous choice for OPTI in kmeans
    #counts = jax.ops.segment_sum(jnp.ones((X.shape[0],)), idxs, num_segments=means.shape[0])
    counts = (idxs[None, :] == jnp.arange(n)[:, None]).sum(axis=1)
    weights = counts / jnp.sum(counts)

    # Estimate the covariance matrices corresponding to these samples
    # OPTI use jnp.where if it is indeed faster
    def estimate_cov(p, m):
        diff = p - m
        return jnp.dot(diff[:,None], diff[None,:])
    covs = jax.ops.segment_sum(jax.vmap(estimate_cov, in_axes=(0,0))(X, means[idxs,:]), idxs, num_segments=means.shape[0])/counts[:,None,None]
    # TODO regularization step
    covs = covs + 1e-6*jnp.eye(means.shape[-1])

    return means, covs, weights



# Initialize and fit a gmm on the data
# TODO doc
@partial(jax.jit, static_argnums=(2,3,4,5))
def gmm(key, X, n, n_try=1, max_iter=100, tol=1e-3):
    means_init, covs_init, weights_init = kmeans_init(key, X, n, n_try, max_iter=max_iter, tol=tol)
    means, covs, weights, _ = gmm_fit(X, means_init, covs_init, weights_init, max_iter=max_iter, tol=tol)
    return means, covs, weights

# prediction function
# TODO doc
@partial(jax.jit)
def predict(X, means, covs, weights):
    dim = covs.shape[1]
    covs_chol = jnp.linalg.cholesky(covs)
    # OPTI check if more efficient to do that or solve X-mu like Aitor did
    # Output of jnp.linalg.cholesky is lower triangular
    inv_covs_chol = jax.vmap(lambda c,e: jscipy.linalg.solve_triangular(c,e,lower=True), in_axes=(0,None))(covs_chol, jnp.eye(dim))
    logpdf = log_pdf(X, means, inv_covs_chol)
    return logpdf
    breakpoint()
    idx = jnp.argmax( + jnp.log(weights[None,:]), axis=-1)
    return idx

##################
def gaussian(X):
    # X is of shape n x f
    print("computing mean and covariance...")
    return jnp.mean(X, axis=0), jnp.linalg.pinv(jnp.cov(X.T))


log2pi = jnp.log(2 * jnp.pi)
def gaussian_predict(X, mus, invsigma):
    """
    Compute the log probabilities
    """
    nsample, nbandes = X.shape
    kchi2 = scipy.stats.chi2(nbandes)
    assert mus.shape == (nbandes,)
    assert invsigma.shape == (nbandes, nbandes)

    # p1 = jnp.log(jscipy.linalg.det(invsigma))  
    # p2 = nbandes * log2pi 
    mahalanobis_dist = ((X - mus[None]) @ invsigma   # nsample x nbandes
     * (X - mus[None])).sum(axis=1)  # nsample
    return kchi2.cdf(mahalanobis_dist)
    # return 0.5 * (p1 - p2 - p3)
