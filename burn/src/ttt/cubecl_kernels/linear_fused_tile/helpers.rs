#![allow(type_alias_bounds)]

use cubecl::prelude::*;
use thundercube::prelude::*;

// Shared memory tiles (St)
pub type CsFTile<P: ParamsTrait> = St<P::E, P::CS, P::F>;
pub type FCsTile<P: ParamsTrait> = St<P::E, P::F, P::CS>;
pub type CsCsTile<P: ParamsTrait> = St<P::E, P::CS, P::CS>;
pub type FFTile<P: ParamsTrait> = St<P::E, P::F, P::F>;

// Shared memory vectors (Sv)
pub type CsVec<P: ParamsTrait> = Sv<P::E, P::CS>;
pub type FVec<P: ParamsTrait> = Sv<P::E, P::F>;

// Register tiles (Rt)
pub type CsCsReg<P: ParamsTrait> = Rt<P::E, P::CS_Reg, P::CS_Reg>;
pub type CsFReg<P: ParamsTrait> = Rt<P::E, P::CS_Reg, P::F_Reg>;
pub type FFReg<P: ParamsTrait> = Rt<P::E, P::F_Reg, P::F_Reg>;

// Register vectors (Rv)
pub type CsRegVec<P: ParamsTrait> = Rv<P::E, P::CS_Reg>;
pub type FRegVec<P: ParamsTrait> = Rv<P::E, P::F_Reg>;
pub type CsRegBig<P: ParamsTrait> = Rv<P::E, P::CS>;
pub type FRegBig<P: ParamsTrait> = Rv<P::E, P::F>;

#[cube]
pub trait ParamsTrait: Send + Sync + 'static {
    type E: Float;
    type CS: Dim;
    type F: Dim;

    type CS_Reg: Dim;
    type F_Reg: Dim;

    // CubeCL won't let us do default impls
    fn cs_f_tile() -> CsFTile<Self>;
    fn f_cs_tile() -> FCsTile<Self>;
    fn cs_cs_tile() -> CsCsTile<Self>;
    fn cs_vec() -> CsVec<Self>;
    fn f_f_tile() -> FFTile<Self>;
    fn f_vec() -> FVec<Self>;

    fn cs_cs_reg() -> CsCsReg<Self>;
    fn cs_f_reg() -> CsFReg<Self>;
    fn f_f_reg() -> FFReg<Self>;
    fn cs_reg() -> CsRegVec<Self>;
    fn f_reg() -> FRegVec<Self>;

    fn cs_reg_big() -> CsRegBig<Self>;
    fn f_reg_big() -> FRegBig<Self>;
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

    fn cs_f_tile() -> CsFTile<Self> {
        St::new()
    }
    fn f_cs_tile() -> FCsTile<Self> {
        St::new()
    }
    fn cs_cs_tile() -> CsCsTile<Self> {
        St::new()
    }
    fn cs_vec() -> CsVec<Self> {
        Sv::new()
    }
    fn f_f_tile() -> FFTile<Self> {
        St::new()
    }
    fn f_vec() -> FVec<Self> {
        Sv::new()
    }

    fn cs_cs_reg() -> CsCsReg<Self> {
        Rt::new()
    }
    fn cs_f_reg() -> CsFReg<Self> {
        Rt::new()
    }
    fn f_f_reg() -> FFReg<Self> {
        Rt::new()
    }

    fn cs_reg() -> CsRegVec<Self> {
        Rv::new()
    }
    fn f_reg() -> FRegVec<Self> {
        Rv::new()
    }

    fn cs_reg_big() -> CsRegBig<Self> {
        Rv::new()
    }
    fn f_reg_big() -> FRegBig<Self> {
        Rv::new()
    }
}

/// Build eta matrix: η[i,j] = token_eta[i] * ttt_lr_eta[j], with triangular mask.
///
/// When `transposed` is false: builds lower triangular η (tril)
/// When `transposed` is true: builds upper triangular η^T (triu)
#[cube]
pub fn build_eta_matrix<P: ParamsTrait>(
    token_eta: &Tensor<Line<P::E>>,
    ttt_lr_eta: &Tensor<Line<P::E>>,
    output: &mut CsCsTile<P>,
    ttt_lr_eta_idx: usize,
    #[comptime] transposed: bool,
) {
    let tiles_per_row = P::CS::VALUE / P::CS_Reg::VALUE;
    let tile_row = (UNIT_POS as usize) / tiles_per_row;
    let tile_col = (UNIT_POS as usize) % tiles_per_row;

    let mut eta_reg = P::cs_cs_reg();
    eta_reg.zero();

    let mut row_vec = P::cs_reg();
    let mut col_vec = P::cs_reg();

    if comptime!(transposed) {
        // η^T[i,j] = ttt_lr_eta[i] * token_eta[j]
        cube::broadcast::load_rv_direct(
            ttt_lr_eta,
            &mut row_vec,
            ttt_lr_eta_idx + tile_row * P::CS_Reg::VALUE,
        );
        cube::broadcast::load_rv_direct(token_eta, &mut col_vec, tile_col * P::CS_Reg::VALUE);
    } else {
        // η[i,j] = token_eta[i] * ttt_lr_eta[j]
        cube::broadcast::load_rv_direct(token_eta, &mut row_vec, tile_row * P::CS_Reg::VALUE);
        cube::broadcast::load_rv_direct(
            ttt_lr_eta,
            &mut col_vec,
            ttt_lr_eta_idx + tile_col * P::CS_Reg::VALUE,
        );
    }

    eta_reg.add_col(&row_vec);
    eta_reg.mul_row(&col_vec);
    cube::store_rt_to_st(&eta_reg, output);

    sync_cube();

    if comptime!(transposed) {
        output.triu();
    } else {
        output.tril();
    }

    sync_cube();
}

/// Compute attention matrix: attn = XQ @ XK^T, with triangular mask.
///
/// When `transposed` is false: builds lower triangular attn (tril)
/// When `transposed` is true: builds upper triangular attn^T (triu)
#[cube]
pub fn build_attn_matrix<P: ParamsTrait>(
    q_smem: &FCsTile<P>,
    k_smem: &FCsTile<P>,
    output: &mut CsCsTile<P>,
    #[comptime] transposed: bool,
) {
    let mut attn_reg = P::cs_cs_reg();
    attn_reg.zero();

    if comptime!(transposed) {
        // attn^T = XK @ XQ^T = k_smem^T @ q_smem
        cube::mma_AtB(&mut attn_reg, k_smem, q_smem);
    } else {
        // attn = XQ @ XK^T = q_smem^T @ k_smem
        cube::mma_AtB(&mut attn_reg, q_smem, k_smem);
    }

    sync_cube();

    cube::store_rt_to_st(&attn_reg, output);

    sync_cube();

    if comptime!(transposed) {
        output.triu();
    } else {
        output.tril();
    }

    sync_cube();
}

/// Compute fused (eta * attn) matrix directly in registers, avoiding separate attn tile.
///
/// Computes: output[i,j] = token_eta[i] * ttt_lr_eta[j] * (q[i] · k[j])
///
/// This fuses build_eta_matrix and build_attn_matrix, computing the element-wise
/// product in registers before storing to shared memory. Saves one CS×CS tile.
#[cube]
pub fn build_eta_attn_fused<P: ParamsTrait>(
    q_smem: &FCsTile<P>,
    k_smem: &FCsTile<P>,
    token_eta: &Tensor<Line<P::E>>,
    ttt_lr_eta: &Tensor<Line<P::E>>,
    output: &mut CsCsTile<P>,
    ttt_lr_eta_idx: usize,
) {
    // Compute attn = q^T @ k in registers
    let mut attn_reg = P::cs_cs_reg();
    attn_reg.zero();
    cube::mma_AtB(&mut attn_reg, q_smem, k_smem);

    sync_cube();

    // Compute eta values and multiply with attn in registers
    let tiles_per_row = P::CS::VALUE / P::CS_Reg::VALUE;
    let tile_row = (UNIT_POS as usize) / tiles_per_row;
    let tile_col = (UNIT_POS as usize) % tiles_per_row;

    // Load eta components: η[i,j] = token_eta[i] * ttt_lr_eta[j]
    let mut row_vec = P::cs_reg();
    let mut col_vec = P::cs_reg();
    cube::broadcast::load_rv_direct(token_eta, &mut row_vec, tile_row * P::CS_Reg::VALUE);
    cube::broadcast::load_rv_direct(
        ttt_lr_eta,
        &mut col_vec,
        ttt_lr_eta_idx + tile_col * P::CS_Reg::VALUE,
    );

    // Build eta in registers and multiply with attn
    let mut eta_reg = P::cs_cs_reg();
    eta_reg.zero();
    eta_reg.add_col(&row_vec);
    eta_reg.mul_row(&col_vec);

    // Element-wise multiply: result = eta * attn
    attn_reg.mul(&eta_reg);

    // Store to shared memory
    cube::store_rt_to_st(&attn_reg, output);

    sync_cube();

    // Apply tril mask
    output.tril();

    sync_cube();
}

// TODO: Move to thundercube and abstract?
/// Extract the last row of a shared memory tile into a register vector.
/// Result is broadcast to all.
#[cube]
#[must_use]
pub fn extract_last_row<F: Float, R: Dim, C: Dim>(st: &St<F, R, C>) -> Rv<F, C> {
    let mut result = Rv::<F, C>::new();
    let last_row = R::VALUE - 1;
    let vec_stride = C::LINES;
    let mask = vec_stride - 1;

    #[unroll]
    for c_line in 0..C::LINES {
        let phys_col = cube::swizzle(last_row, c_line, mask);
        let s_idx = last_row * vec_stride + phys_col;
        result.data[c_line] = st.data[s_idx];
    }
    result
}
