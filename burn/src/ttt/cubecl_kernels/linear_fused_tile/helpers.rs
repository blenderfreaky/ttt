use cubecl::prelude::*;
use thundercube::prelude::*;

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
    fn f_f_reg() -> Rt<Self::E, Self::F_Reg, Self::F_Reg>;
    fn cs_reg() -> Rv<Self::E, Self::CS_Reg>;
    fn f_reg() -> Rv<Self::E, Self::F_Reg>;

    fn cs_reg_big() -> Rv<Self::E, Self::CS>;
    fn f_reg_big() -> Rv<Self::E, Self::F>;
}

// pub type CS_F_Tile<P: ParamsTrait> = St<P::E, P::CS, P::F>;
// pub type F_CS_Tile<P: ParamsTrait> = St<P::E, P::F, P::CS>;
// pub type CS_CS_Tile<P: ParamsTrait> = St<P::E, P::CS, P::CS>;
// pub type CS_F_Reg<P: ParamsTrait> = Rt<P::E, P::CS_Reg, P::F_Reg>;
// pub type F_CS_Reg<P: ParamsTrait> = Rt<P::E, P::F_Reg, P::CS_Reg>;
// pub type CS_CS_Reg<P: ParamsTrait> = CS_CS_Tile<P>;

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
    fn f_f_reg() -> Rt<Self::E, Self::F_Reg, Self::F_Reg> {
        Rt::new()
    }

    fn cs_reg() -> Rv<Self::E, Self::CS_Reg> {
        Rv::new()
    }
    fn f_reg() -> Rv<Self::E, Self::F_Reg> {
        Rv::new()
    }

    fn cs_reg_big() -> Rv<Self::E, Self::CS> {
        Rv::new()
    }
    fn f_reg_big() -> Rv<Self::E, Self::F> {
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
    output: &mut St<P::E, P::CS, P::CS>,
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
    q_smem: &St<P::E, P::F, P::CS>,
    k_smem: &St<P::E, P::F, P::CS>,
    output: &mut St<P::E, P::CS, P::CS>,
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
