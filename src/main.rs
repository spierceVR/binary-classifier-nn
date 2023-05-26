pub mod neural_network;
pub mod neuron;
pub mod utils;

use ndarray::array;
use neural_network::MyNeuralNetwork;

enum Gender {
    Male,
    Female,
}

fn gender_as_float(gen:&Gender) -> f32 {
    match gen {
        Gender::Male => 0.0,
        Gender::Female => 1.0,
    }
}

fn main() {
    // 135lbs 5'9-> 0
    let x = array![0.0, 3.0]; // [weight - 135lbs, height - 66in]
    // 160lbs 6'2 -> 0
    let x1 = array![25.0, 8.0]; // [weight - 135lbs, height - 66in]
    // 110lbs 5'6 -> 1
    let x2 = array![-15.0, 0.0]; // [weight - 135lbs, height - 66in]
    // 100lbs 5'2 -> 1
    let x3 = array![-35.0, -4.0]; // [weight - 135lbs, height - 66in]

    let mut net = MyNeuralNetwork::new();
    let epochs = 1000;
    let learning_rate = 0.1;

    let data = vec![(x, 0.0), (x1, 0.0), (x2, 1.0), (x3, 1.0)]; // (x, y_actual)
    net.train(&data, epochs, learning_rate);

    // make a prediction for a new input ( 115 lbs, 5'4") ( Female )
    let y0_pred = net.predict(&array![-20.0, -2.0]); 
    println!("y_pred: {}", y0_pred); // should be close to 1.0
    println!("y_actual: {}", gender_as_float(&Gender::Female)); // 1.0


}
