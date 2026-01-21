#![allow(non_camel_case_types, non_snake_case)]
#![allow(dead_code)]

use cubecl::prelude::*;
use thundercube::{prelude::*, util::index_2d};

use crate::ttt::cubecl_kernels::FusedTttConfig;

#[derive(CubeType, CubeLaunch)]
pub struct Inputs<F: Float> {
    pub xq: Tensor<Line<F>>,
    pub xk: Tensor<Line<F>>,
    pub xv: Tensor<Line<F>>,
    pub weight: Tensor<Line<F>>,
    pub bias: Tensor<Line<F>>,
    pub token_eta: Tensor<Line<F>>,
    pub ttt_lr_eta: Tensor<Line<F>>,
    pub ln_weight: Tensor<Line<F>>,
    pub ln_bias: Tensor<Line<F>>,
}

// We use a trait here because
// inherent assoc types are unstable
#[cube]
pub trait ParamsTrait: Send + Sync + 'static {
    type E: Float;
    type CS: Dim;
    type F: Dim;

    type CS_Reg: Dim;
    type F_Reg: Dim;

    // CubeCL won't let us do default impls
    fn cs_f_tile() -> St<Self::E, Self::CS, Self::F>;
    fn f_cs_tile() -> St<Self::E, Self::F, Self::CS>;
    fn cs_cs_tile() -> St<Self::E, Self::CS, Self::CS>;
    fn cs_vec() -> Sv<Self::E, Self::CS>;
    fn f_f_tile() -> St<Self::E, Self::F, Self::F>;
    fn f_vec() -> Sv<Self::E, Self::F>;

    fn cs_cs_reg() -> Rt<Self::E, Self::CS_Reg, Self::CS_Reg>;
    fn cs_f_reg() -> Rt<Self::E, Self::CS_Reg, Self::F_Reg>;
    fn cs_reg() -> Rv<Self::E, Self::CS_Reg>;
}

pub struct Params<E: Float, CS: Dim, F: Dim, CS_Reg: Dim, F_Reg: Dim> {
    _phantom: std::marker::PhantomData<(E, CS, F, CS_Reg, F_Reg)>,
}

#[cube]
impl<E: Float, CS: Dim, F: Dim, CS_Reg: Dim, F_Reg: Dim> ParamsTrait
    for Params<E, CS, F, CS_Reg, F_Reg>
{
    type E = E;
    type CS = CS;
    type F = F;
    type CS_Reg = CS_Reg;
    type F_Reg = F_Reg;

    fn cs_f_tile() -> St<Self::E, Self::CS, Self::F> {
        St::new()
    }
    fn f_cs_tile() -> St<Self::E, Self::F, Self::CS> {
        St::new()
    }
    fn cs_cs_tile() -> St<Self::E, Self::CS, Self::CS> {
        St::new()
    }
    fn cs_vec() -> Sv<Self::E, Self::CS> {
        Sv::new()
    }
    fn f_f_tile() -> St<Self::E, Self::F, Self::F> {
        St::new()
    }
    fn f_vec() -> Sv<Self::E, Self::F> {
        Sv::new()
    }

    fn cs_cs_reg() -> Rt<Self::E, Self::CS_Reg, Self::CS_Reg> {
        Rt::new()
    }
    fn cs_f_reg() -> Rt<Self::E, Self::CS_Reg, Self::F_Reg> {
        Rt::new()
    }

    fn cs_reg() -> Rv<Self::E, Self::CS_Reg> {
        Rv::new()
    }
}

/// Fused TTT-Linear backward pass kernel.
///
/// Each CUBE handles one (batch, head) pair.
/// Thread layout: (head_dim, seq_len)
#[cube(launch)]
pub fn fused_ttt_backward_kernel<P: ParamsTrait>(
    inputs: Inputs<P::E>,
    #[comptime] config: FusedTttConfig,
) {
    let B = inputs.xq.shape(0);
    let NH = inputs.xq.shape(1);
    let CS = P::CS::VALUE;
    let F = P::F::VALUE;

    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;

    let epsilon = comptime!(config.epsilon());

    let mut weight_smem = P::f_f_tile();
    let mut bias_smem = P::f_vec();

    let mut q_smem = P::f_cs_tile();
    let mut k_smem = P::f_cs_tile();
    let mut v_smem = P::f_cs_tile();

    let mut token_eta_smem = P::cs_vec();
    let mut ttt_lr_eta_smem = P::cs_vec();

    let mut ln_weight_smem = P::f_vec();
    let mut ln_bias_smem = P::f_vec();

    let mut z1_smem = P::cs_f_tile();

    let mut grad_l_wrt_z1_smem = P::cs_f_tile();

    let mut bias_bar_smem = P::cs_f_tile();
    let mut z1_bar_smem = P::cs_f_tile();

    let mut eta_matrix_smem = P::cs_cs_tile();

    // // why 16?
    // let mut ln_smem = Sv::<P::E, D16>::new();

    // let mut matmul_smem = F::cs_f_tile();
    // let mut b_acc_smem = F::cs_f_tile();

    plane::load_st_direct(
        &inputs.weight,
        &mut weight_smem,
        index_2d(&inputs.weight, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_direct(
        &inputs.bias,
        &mut bias_smem,
        index_2d(&inputs.bias, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_transpose(
        &inputs.xq,
        &mut q_smem,
        index_2d(&inputs.xq, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_transpose(
        &inputs.xk,
        &mut k_smem,
        index_2d(&inputs.xk, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_transpose(
        &inputs.xv,
        &mut v_smem,
        index_2d(&inputs.xv, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_direct(
        &inputs.token_eta,
        &mut token_eta_smem,
        index_2d(&inputs.token_eta, 0, 0),
        0,
        0,
    );
    plane::load_st_direct(
        &inputs.ttt_lr_eta,
        &mut ttt_lr_eta_smem,
        index_2d(&inputs.ttt_lr_eta, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_direct(
        &inputs.ln_weight,
        &mut ln_weight_smem,
        index_2d(&inputs.ln_weight, 0, head_idx),
        0,
        0,
    );
    plane::load_st_direct(
        &inputs.ln_bias,
        &mut ln_bias_smem,
        index_2d(&inputs.ln_bias, 0, head_idx),
        0,
        0,
    );

    sync_plane();

    let cs_cs_reg = P::cs_cs_reg();
    let cs_f_reg = P::cs_f_reg();
    // let mut cs_cs_reg_2 = P::cs_cs_reg();
    let cs_reg = P::cs_reg();

    // z1 = x1 @ W + b
    let mut z1_reg = cs_cs_reg;
    z1_reg.zero();
    plane::mma_AtB(&mut z1_reg, &k_smem, &weight_smem);

    let mut bias_reg = cs_reg;
    plane::load_rt_from_st(&bias_smem, &mut bias_reg);
    z1_reg.add_col(&bias_reg);
    let cs_reg = bias_reg;
    plane::store_rt_to_st(&z1_reg, &mut z1_smem);
    let cs_cs_reg = z1_reg;

    // reconstruction_target = v - k
    v_smem.sub(&k_smem);
    let reconstruction_target = v_smem;

    // TODO: ln grad

    let mut eta_matrix_reg = cs_cs_reg;

    eta_matrix_reg.zero();
    let mut token_eta_reg = cs_reg;
    plane::load_rt_from_st(&token_eta_smem, &mut token_eta_reg);
    // TODO: May be transposed
    eta_matrix_reg.add_row(&token_eta_reg);
    let cs_reg = token_eta_reg;

    let mut ttt_lr_eta_reg = cs_reg;
    plane::load_rt_from_st(&ttt_lr_eta_smem, &mut ttt_lr_eta_reg);
    // TODO: May be transposed
    eta_matrix_reg.mul_col(&ttt_lr_eta_reg);
    let cs_reg = ttt_lr_eta_reg;

    plane::store_rt_to_st(&eta_matrix_reg, &mut eta_matrix_smem);

    // TODO: eta_matrix.tril

    let mut attn_scores = P::cs_cs_reg();

    // TODO, may need a transpose on one side
    plane::mma_AtB(&mut attn_scores, &q_smem, &k_smem);

    // TODO: attn_scores.tril

    let mut bias_bar_reg = cs_f_reg;
    bias_bar_reg.zero();
    plane::mma_AtB(&mut bias_bar_reg, &eta_matrix_smem, &grad_l_wrt_z1_smem);
    bias_bar_reg.neg();

    let mut bias_reg = cs_reg;
    plane::load_rt_from_st(&bias_smem, &mut bias_reg);
    bias_bar_reg.add_col(&bias_reg);
    let cs_reg = bias_reg;

    plane::store_rt_to_st(&bias_bar_reg, &mut bias_bar_smem);
    let cs_f_reg = bias_bar_reg;

    // let mut z1_bar_reg = cs_f_reg;
}
