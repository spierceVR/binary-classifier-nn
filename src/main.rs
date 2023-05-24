pub mod neural_network;
pub mod neuron;
pub mod utils;

use ndarray::array;
use neural_network::MyNeuralNetwork;

enum Gender {
    Male,
    Female,
}

fn genderAsFloat(gen: Gender) -> f32 {
    match gen {
        Male => 1.0,
        Female => 0.0,
    }
}

fn main() {
    // 135lbs 5'9-> 1
    let x = array![0.0, 3.0]; // [weight - 135lbs, height - 66in]
    // 160lbs 6'2 -> 1
    let x1 = array![25.0, 8.0]; // [weight - 135lbs, height - 66in]
    // 110lbs 5'6 -> 0
    let x2 = array![-15.0, 0.0]; // [weight - 135lbs, height - 66in]
    // 100lbs 5'2 -> 0
    let x3 = array![-35.0, -4.0]; // [weight - 135lbs, height - 66in]

    let mut net = MyNeuralNetwork::new();
    let epochs = 1000;
    let learning_rate = 0.1;
    let y_actual = genderAsFloat(Gender::Male);

    let data = vec![(x, 1.0)]; // (x, y_actual)
    net.train(data, epochs, learning_rate);

    let y_pred = net.predict(array![2.0, 3.0]);


}
