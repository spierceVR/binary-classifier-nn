#![allow(non_snake_case)]
use ndarray::Array1;
use ndarray::array;
use rand::thread_rng;
use rand::Rng;
use crate::neuron::Neuron;
use crate::utils::math::deriv_sigmoid;
use crate::utils::math::sigmoid;
pub(crate) struct MyNeuralNetwork {
    //   A neural network with:
    //     - 2 inputs ( weight and height )
    //     - a hidden layer with 2 neurons (h1, h2)
    //     - an output layer with 1 neuron (o1) - gender
    //   Each neuron has the same weights and bias:
    //     - w = [0, 1]
    //     - b = 0
    h1: Neuron,
    h2: Neuron,
    o: Neuron,
}

impl MyNeuralNetwork {
    pub(crate) fn new() -> Self {
        let h1: Neuron = Neuron::new(&array![thread_rng().gen::<f32>(), thread_rng().gen::<f32>()], thread_rng().gen::<f32>());
        let h2: Neuron = Neuron::new(&array![thread_rng().gen::<f32>(), thread_rng().gen::<f32>()], thread_rng().gen::<f32>());
        let o: Neuron = Neuron::new(&array![thread_rng().gen::<f32>(), thread_rng().gen::<f32>()], thread_rng().gen::<f32>());

        Self { h1, h2, o }
    }

    fn feedforward(&self, inputs: &Array1<f32>) -> f32 {
        let h1o = self.h1.feedforward(inputs);
        let h2o = self.h2.feedforward(inputs);
        self.o.feedforward(&array![h1o, h2o])
    }

    fn mse_loss(y_actual: &Array1<f32>, y_pred: &Array1<f32>) -> f32 {
        let diff = y_actual - y_pred;
        diff.mapv(|x: f32| -> f32 {x.powi(2)}).mean().unwrap()
    }

    pub(crate) fn predict(&self, inputs: &Array1<f32>) -> f32 {
        self.feedforward(inputs)
    }
    
    fn partial_L_with_respect_to_yPred(y_actual: f32, y_pred: f32) -> f32 {
        -2.0 * (y_actual - y_pred)
    }

    pub(crate) fn train(
        &mut self,
        data: &Vec<(Array1<f32>, f32)>,
        epochs: i32,
        learning_rate: f32,
    ) {
        // input shape is (2)
        assert_eq!(data[0].0.shape(), array![0,0].shape());
        // data is a vec of tuples (input, y_true)
        // input is an ndarray of shape (2,)
        // y_actual is a float 

        for epoch in 0..epochs {
            for (x, y_actual) in data {
                // Feedforward
                // # Math: sum_{h1} = w_1 * x_1 + w_2 * x_2 + b
                let sum_h1 =
                    self.h1.weights()[0] * x[0] + self.h1.weights()[1] * x[1] + self.h1.bias();
                // # Math: h1 = sigmoid(sum_{h1})
                let h1 = sigmoid(sum_h1);
                // # Math: sum_{h2} = w_1 * x_1 + w_2 * x_2 + b
                let sum_h2 =
                    self.h2.weights()[0] * x[0] + self.h2.weights()[1] * x[1] + self.h2.bias();
                // # Math: h2 = sigmoid(sum_{h2})
                let h2 = sigmoid(sum_h2);
                // # Math: sum_{o1} = w_1 * h1 + w_2 * h2 + b
                let sum_o = self.o.weights()[0] * h1 + self.o.weights()[1] * h2 + self.o.bias();
                // # Math: o1 = sigmoid(sum_{o1})
                let o = sigmoid(sum_o);
                

                // Calculate partial derivative of loss with respect to each weight and bias
                // using chain rule, partial L / partial wj = (partial L by partial y_pred) * (partial y_pred by partial wi)
                // # Math: \frac{\partial{L_i}}{\partial{w_j}} = \frac{\partial{L_i}}{\partial{y_{pred_i}}} * \frac{\partial{y_{pred_i}}}{\partial{w_j}} 
                // the first multiplicand is common to all weights and biases, so we calculate it first
                let dL_dO = MyNeuralNetwork::partial_L_with_respect_to_yPred(*y_actual, o);
                // NOTE: dL_dO is the same for all weights and biases of the output neuron


                // for each weight and bias, calculate the second multiplicand for each weight and bias
                // # Math: \frac{\partial{y_{pred_i}}}{\partial{w_j}} = \frac{\partial{y_{pred_i}}}{\partial{sum_{neuron(w_j)}}} * \frac{\partial{sum_{neuron(w_j)}}}{\partial{w_j}}
                // # Math: \frac{\partial{y_{pred_i}}}{\partial{sum_{i}}} = \frac{\partial{sigmoid(sum_{i})}}{\partial{sum_{i}}} = sigmoid'(sum_{i})
               
                // Backpropagation

                // # Math: \frac{\partial{y_{pred}}} {\partial{h1}} = \frac{\partial{sigmoid(sum_{o1})}} {\partial{h1}} =  0.w_1 * sigmoid'(sum_{o1})
                let dO_dh1 = self.o.weights()[0] * deriv_sigmoid(sum_o);
                let dh1_dw5 = h1;
                let dO_dw5 = dO_dh1 * dh1_dw5;
                // # Math: \frac{\partial{y_{pred}}} {\partial{h2}} = \frac{\partial{sigmoid(sum_{o1})}} {\partial{h2}} =  0.w_2 * sigmoid'(sum_{o1})
                let dO_dh2 = self.o.weights()[1] * deriv_sigmoid(sum_o);
                let dh2_dw6 = h2;
                let dO_dw6 = dO_dh2 * dh2_dw6;
                // # Math: \frac{\partial{y_{pred}}} {\partial{b_3}} = \frac{\partial{sigmoid(sum_{o1})}} {\partial{b_3}} =  1 * sigmoid'(sum_{o1})
                let dO_db3 = deriv_sigmoid(sum_o);
                * (self.o.mut_bias()) = self.o.bias() - learning_rate * dL_dO * dO_db3;
                *(self.o.mut_weights()) = array![
                    // # Math: \frac{\partial{y_{pred}}} {\partial{w_5}} = \frac{\partial{sigmoid(sum_{o1})}} {\partial{w_5}} =  h1 * sigmoid'(sum_{o1})
                    self.o.weights()[0] - learning_rate * dL_dO * dO_dw5,
                    // # Math: \frac{\partial{y_{pred}}} {\partial{w_6}} = \frac{\partial{sigmoid(sum_{o1})}} {\partial{w_6}} =  h2 * sigmoid'(sum_{o1})
                    self.o.weights()[1] - learning_rate * dL_dO * dO_dw6,
                ];
                *(self.o.mut_bias()) = self.o.bias() - learning_rate * dL_dO * dO_db3;
                let dO_dh1 = self.o.weights()[0] * deriv_sigmoid(sum_o);
                let dO_dh2 = self.o.weights()[1] * deriv_sigmoid(sum_o);
                let dh1_dw1 = x[0];
                let dh1_dw2 = x[1];
                let dh2_dw3 = x[0];
                let dh2_dw4 = x[1];
                *(self.h1.mut_weights()) = array![
                    // # Math: \frac{\partial{sum_{o1}}} {\partial{h1}} = \frac{\partial{0.w_1 * h1 + w_6 * h2 + b_3}} {\partial{h1}} = 0.w_1
                    self.h1.weights()[0] - learning_rate  * dL_dO * dO_dh1 * dh1_dw1,
                    self.h1.weights()[1] - learning_rate  * dL_dO * dO_dh1 * dh1_dw2,
                ];
                *(self.h2.mut_weights()) = array![
                    // # Math: \frac{\partial{sum_{o1}}} {\partial{h2}} = \frac{\partial{0.w_1 * h1 + w_6 * h2 + b_3}} {\partial{h2}} = 0.w_2
                    self.h2.weights()[0] - learning_rate  * dL_dO * dO_dh2 * dh2_dw3,
                    self.h2.weights()[1] - learning_rate  * dL_dO * dO_dh2 * dh2_dw4,
                ];

                // Calculate overall loss
                let y_pred = o;
                if x == data[0].0 { // once per epoch
                    let loss = MyNeuralNetwork::mse_loss(&array![*y_actual], &array![y_pred]);
                    println!("Epoch {}: Loss = {}", epoch, loss);
                }
            }
        }
    }

}
