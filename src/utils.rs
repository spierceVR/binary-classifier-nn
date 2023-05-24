pub mod math {

    pub(crate) fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub(crate) fn deriv_sigmoid(x: f32) -> f32 {
        let fx = sigmoid(x);
        fx * (1.0 - fx)
    }
}
