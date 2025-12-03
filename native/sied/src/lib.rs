//! # sied - SIMD operations for Erlang
//!
//! This crate provides high-performance vectorized operations for Erlang
//! through Rustler NIFs. All implementations use simdeez for portable
//! SIMD abstractions with runtime dispatch.
//!
//! ## Features
//! - Runtime SIMD detection (SSE2, SSE4.1, AVX2, NEON)
//! - Comprehensive mathematical operations
//! - Safe Rust with automatic SIMD optimization
//! - Zero unsafe code
use rustler::{Encoder, Env, Error, Term};
use simdeez::prelude::*;

/// Atom definitions for consistent error and success responses
mod atoms {
    rustler::atoms! {
        ok,
        error,
        length_mismatch,
        invalid_input,
        empty_vector
    }
}

rustler::init!("sied");

//==============================================================================
// Basic Arithmetic Operations
//==============================================================================
/// Element-wise addition of two f32 vectors
#[rustler::nif]
fn add_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = add_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise addition of two f64 vectors
#[rustler::nif]
fn add_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = add_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise subtraction of two f32 vectors
#[rustler::nif]
fn subtract_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = subtract_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise subtraction of two f64 vectors
#[rustler::nif]
fn subtract_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = subtract_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise multiplication of two f32 vectors
#[rustler::nif]
fn multiply_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = multiply_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise multiplication of two f64 vectors
#[rustler::nif]
fn multiply_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = multiply_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise division of two f32 vectors
#[rustler::nif]
fn divide_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = divide_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise division of two f64 vectors
#[rustler::nif]
fn divide_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = divide_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

//==============================================================================
// Reduction Operations
//==============================================================================
/// Dot product of two f32 vectors
#[rustler::nif]
fn dot_product_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = dot_product_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Dot product of two f64 vectors
#[rustler::nif]
fn dot_product_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = dot_product_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Sum of all elements in an f32 vector
#[rustler::nif]
fn sum_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    let result = sum_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Sum of all elements in an f64 vector
#[rustler::nif]
fn sum_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    let result = sum_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

//==============================================================================
// Statistical Operations
//==============================================================================
/// Mean (average) of an f32 vector
#[rustler::nif]
fn mean_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = sum_f32_simd(&a) / a.len() as f32;
    Ok((atoms::ok(), result).encode(env))
}

/// Mean (average) of an f64 vector
#[rustler::nif]
fn mean_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = sum_f64_simd(&a) / a.len() as f64;
    Ok((atoms::ok(), result).encode(env))
}

/// Variance of an f32 vector
#[rustler::nif]
fn variance_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let mean = sum_f32_simd(&a) / a.len() as f32;
    let mean_vec = vec![mean; a.len()];
    let diff = subtract_f32_simd(&a, &mean_vec);
    let squared = multiply_f32_simd(&diff, &diff);
    let result = sum_f32_simd(&squared) / a.len() as f32;
    Ok((atoms::ok(), result).encode(env))
}

/// Variance of an f64 vector
#[rustler::nif]
fn variance_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let mean = sum_f64_simd(&a) / a.len() as f64;
    let mean_vec = vec![mean; a.len()];
    let diff = subtract_f64_simd(&a, &mean_vec);
    let squared = multiply_f64_simd(&diff, &diff);
    let result = sum_f64_simd(&squared) / a.len() as f64;
    Ok((atoms::ok(), result).encode(env))
}

/// Standard deviation of an f32 vector
#[rustler::nif]
fn std_dev_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let mean = sum_f32_simd(&a) / a.len() as f32;
    let mean_vec = vec![mean; a.len()];
    let diff = subtract_f32_simd(&a, &mean_vec);
    let squared = multiply_f32_simd(&diff, &diff);
    let variance = sum_f32_simd(&squared) / a.len() as f32;
    let result = variance.sqrt();
    Ok((atoms::ok(), result).encode(env))
}

/// Standard deviation of an f64 vector
#[rustler::nif]
fn std_dev_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let mean = sum_f64_simd(&a) / a.len() as f64;
    let mean_vec = vec![mean; a.len()];
    let diff = subtract_f64_simd(&a, &mean_vec);
    let squared = multiply_f64_simd(&diff, &diff);
    let variance = sum_f64_simd(&squared) / a.len() as f64;
    let result = variance.sqrt();
    Ok((atoms::ok(), result).encode(env))
}

//==============================================================================
// Min/Max Operations
//==============================================================================
/// Minimum value in an f32 vector
#[rustler::nif]
fn min_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = min_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Minimum value in an f64 vector
#[rustler::nif]
fn min_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = min_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Maximum value in an f32 vector
#[rustler::nif]
fn max_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = max_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Maximum value in an f64 vector
#[rustler::nif]
fn max_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    if a.is_empty() {
        return Ok((atoms::error(), atoms::empty_vector()).encode(env));
    }
    let result = max_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise minimum of two f32 vectors
#[rustler::nif]
fn min_elementwise_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = min_elementwise_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise minimum of two f64 vectors
#[rustler::nif]
fn min_elementwise_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = min_elementwise_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise maximum of two f32 vectors
#[rustler::nif]
fn max_elementwise_f32(env: Env, a: Vec<f32>, b: Vec<f32>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = max_elementwise_f32_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

/// Element-wise maximum of two f64 vectors
#[rustler::nif]
fn max_elementwise_f64(env: Env, a: Vec<f64>, b: Vec<f64>) -> Result<Term, Error> {
    if a.len() != b.len() {
        return Ok((atoms::error(), atoms::length_mismatch()).encode(env));
    }
    let result = max_elementwise_f64_simd(&a, &b);
    Ok((atoms::ok(), result).encode(env))
}

//==============================================================================
// Unary Operations
//==============================================================================
/// Absolute value of an f32 vector
#[rustler::nif]
fn abs_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    let result = abs_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Absolute value of an f64 vector
#[rustler::nif]
fn abs_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    let result = abs_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Square root of an f32 vector
#[rustler::nif]
fn sqrt_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    let result = sqrt_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Square root of an f64 vector
#[rustler::nif]
fn sqrt_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    let result = sqrt_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Negate an f32 vector
#[rustler::nif]
fn negate_f32(env: Env, a: Vec<f32>) -> Result<Term, Error> {
    let result = negate_f32_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

/// Negate an f64 vector
#[rustler::nif]
fn negate_f64(env: Env, a: Vec<f64>) -> Result<Term, Error> {
    let result = negate_f64_simd(&a);
    Ok((atoms::ok(), result).encode(env))
}

//==============================================================================
// SIMD Implementation Functions using simdeez
//==============================================================================
simd_runtime_generate! {
    fn add_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va + vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] + b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn add_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va + vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] + b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn subtract_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va - vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] - b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn subtract_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va - vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] - b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn multiply_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va * vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] * b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn multiply_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va * vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] * b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn divide_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va / vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] / b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn divide_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va / vb;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i] / b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn dot_product_f32_simd(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = S::Vf32::set1(0.0);
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            sum = sum + (va * vb);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
        }
        let mut result = sum.horizontal_add();
        for i in 0..a_slice.len() {
            result += a_slice[i] * b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn dot_product_f64_simd(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = S::Vf64::set1(0.0);
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            sum = sum + (va * vb);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
        }
        let mut result = sum.horizontal_add();
        for i in 0..a_slice.len() {
            result += a_slice[i] * b_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn sum_f32_simd(a: &[f32]) -> f32 {
        let mut sum = S::Vf32::set1(0.0);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            sum = sum + va;
            a_slice = &a_slice[S::Vf32::WIDTH..];
        }
        let mut result = sum.horizontal_add();
        for i in 0..a_slice.len() {
            result += a_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn sum_f64_simd(a: &[f64]) -> f64 {
        let mut sum = S::Vf64::set1(0.0);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            sum = sum + va;
            a_slice = &a_slice[S::Vf64::WIDTH..];
        }
        let mut result = sum.horizontal_add();
        for i in 0..a_slice.len() {
            result += a_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn min_f32_simd(a: &[f32]) -> f32 {
        let mut min_val = S::Vf32::set1(f32::INFINITY);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            min_val = min_val.min(va);
            a_slice = &a_slice[S::Vf32::WIDTH..];
        }
        let mut result = min_val.horizontal_add();
        for &val in a_slice {
            if val < result {
                result = val;
            }
        }
        result
    }
}

simd_runtime_generate! {
    fn min_f64_simd(a: &[f64]) -> f64 {
        let mut min_val = S::Vf64::set1(f64::INFINITY);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            min_val = min_val.min(va);
            a_slice = &a_slice[S::Vf64::WIDTH..];
        }
        let mut result = min_val.horizontal_add();
        for &val in a_slice {
            if val < result {
                result = val;
            }
        }
        result
    }
}

simd_runtime_generate! {
    fn max_f32_simd(a: &[f32]) -> f32 {
        let mut max_val = S::Vf32::set1(f32::NEG_INFINITY);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            max_val = max_val.max(va);
            a_slice = &a_slice[S::Vf32::WIDTH..];
        }
        let mut result = max_val.horizontal_add();
        for &val in a_slice {
            if val > result {
                result = val;
            }
        }
        result
    }
}

simd_runtime_generate! {
    fn max_f64_simd(a: &[f64]) -> f64 {
        let mut max_val = S::Vf64::set1(f64::NEG_INFINITY);
        let mut a_slice = &a[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            max_val = max_val.max(va);
            a_slice = &a_slice[S::Vf64::WIDTH..];
        }
        let mut result = max_val.horizontal_add();
        for &val in a_slice {
            if val > result {
                result = val;
            }
        }
        result
    }
}

simd_runtime_generate! {
    fn min_elementwise_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va.min(vb);
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].min(b_slice[i]);
        }
        result
    }
}

simd_runtime_generate! {
    fn min_elementwise_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va.min(vb);
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].min(b_slice[i]);
        }
        result
    }
}

simd_runtime_generate! {
    fn max_elementwise_f32_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vb = S::Vf32::load_from_slice(b_slice);
            let vr = va.max(vb);
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            b_slice = &b_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].max(b_slice[i]);
        }
        result
    }
}

simd_runtime_generate! {
    fn max_elementwise_f64_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut b_slice = &b[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vb = S::Vf64::load_from_slice(b_slice);
            let vr = va.max(vb);
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            b_slice = &b_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].max(b_slice[i]);
        }
        result
    }
}

simd_runtime_generate! {
    fn abs_f32_simd(a: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vr = va.abs();
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].abs();
        }
        result
    }
}

simd_runtime_generate! {
    fn abs_f64_simd(a: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vr = va.abs();
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].abs();
        }
        result
    }
}

simd_runtime_generate! {
    fn sqrt_f32_simd(a: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vr = va.sqrt();
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].sqrt();
        }
        result
    }
}

simd_runtime_generate! {
    fn sqrt_f64_simd(a: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vr = va.sqrt();
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = a_slice[i].sqrt();
        }
        result
    }
}

simd_runtime_generate! {
    fn negate_f32_simd(a: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let zero = S::Vf32::set1(0.0);
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(a_slice);
            let vr = zero - va;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf32::WIDTH..];
            res_slice = &mut res_slice[S::Vf32::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = -a_slice[i];
        }
        result
    }
}

simd_runtime_generate! {
    fn negate_f64_simd(a: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(a.len());
        unsafe { result.set_len(a.len()); }
        let zero = S::Vf64::set1(0.0);
        let mut a_slice = &a[..];
        let mut res_slice = &mut result[..];
        while a_slice.len() >= S::Vf64::WIDTH {
            let va = S::Vf64::load_from_slice(a_slice);
            let vr = zero - va;
            vr.copy_to_slice(res_slice);
            a_slice = &a_slice[S::Vf64::WIDTH..];
            res_slice = &mut res_slice[S::Vf64::WIDTH..];
        }
        for i in 0..a_slice.len() {
            res_slice[i] = -a_slice[i];
        }
        result
    }
}

