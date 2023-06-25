use std::{cell::RefCell, borrow::BorrowMut, ops::{Mul, AddAssign}};

use rand::Rng;
use ndarray::{arr2, arr1, Array1, Array2, Array, Axis};

#[derive(Debug, Default)]
pub struct Layer {
    value_a: RefCell<Array1<f32>>,
    unscaled_z: RefCell<Array1<f32>>,
    error_z: RefCell<Array1<f32>>,
    pub bias_b: RefCell<Array1<f32>>,
    gradient_b: RefCell<Array1<f32>>
}

impl Layer {
    pub fn len(&self) -> usize {
        self.value_a.borrow().len()
    }
}

pub type Input = Vec<f32>;
pub type Output = Vec<f32>;
pub struct Sample {
    pub input: Input,
    pub output: Output,
}

pub type Samples = Vec<Sample>;

fn new_vec(size: usize) -> Array1<f32> {
    let mut vec = Vec::with_capacity(size);

    for _ in 0..size  {
        vec.push(0.0);
    }

    arr1(&vec)
}


fn new_layer(size: usize) -> Layer {
    let mut rng = rand::thread_rng();
    
    let layer = Layer {
        value_a: RefCell::new(new_vec(size)),
        unscaled_z: RefCell::new(new_vec(size)),
        error_z: RefCell::new(new_vec(size)),
        bias_b: RefCell::new(new_vec(size)),
        gradient_b: RefCell::new(new_vec(size)),
    };

    // Init each neuron with a random bias
    let mut bias = layer.bias_b.borrow_mut();
    for i in 0..size  {
        bias[i] = rng.gen_range(-1.0..1.0);
    }

    drop(bias);

    layer
}

#[derive(Debug)]
pub struct Connections {
    pub value_w: Array2<f32>,
    gradient_w: Array2<f32>,
}

fn new_connections(current: &Layer, previous: &Layer) -> Connections {
    let mut rng = rand::thread_rng();

    let mut value = Array::<f32, _>::zeros((current.len(), previous.len()));
    let gradient = Array::<f32, _>::zeros((current.len(), previous.len()));

    // Init each connection with a random weight
    for i in 0..current.len() {
        for j in 0..previous.len() {   
            value[(i, j)] = rng.gen_range(-1.0..1.0);                
        }
    }

    Connections {
        value_w: value,
        gradient_w: gradient
    }
}

#[derive(Debug)]
pub struct NN {
    pub layers: Vec<Layer>,
    pub connections: Vec<Connections>
}

impl NN {
    pub fn new(arch: &[usize]) -> Box<Self> {
        assert!(arch.len() > 1, "Invalid arch");


        // create each layer
        let mut layers = Vec::with_capacity(arch.len());

        for layer in 0..arch.len() {
            layers.push(new_layer(arch[layer]))
        }


        // create each connection, connections are the ammount of layers - 1
        let mut connections = Vec::with_capacity(arch.len() - 1);

        for connection in 1..arch.len() {
            connections.push(new_connections(&layers[connection], &layers[connection - 1]))
        }


        // compose layers and connections
        Box::new(Self {
            layers,
            connections
        })
    }

    // let out = io[0] as f32 * nn.wa + io[1] as f32 * nn.wb + nn.b;
    // fn forward_old(&mut self, input: &Input) {

    //     let layer0 = self.layers[0].borrow_mut();

    //     // layer 0 is input
    //     assert_eq!(input.len(), layer0.len(), "Input layers not of same size!");

    //     // could probably swap layer 1 for input layer?
    //     let mut value_a = layer0.value_a.borrow_mut();
    //     for i in 0..input.len() {
    //         value_a[i] = input[i]
    //     }

    //     drop(value_a);

    //     // calculate value for each successive layer
    //     for curr_layer_i in 1..self.layers.len() {

    //         let connection = &self.connections[curr_layer_i - 1];

    //         let mut curr_unscaled_z = self.layers[curr_layer_i].unscaled_z.borrow_mut();
    //         let mut curr_value_a = self.layers[curr_layer_i].value_a.borrow_mut();


    //         let prev_bias_b = self.layers[curr_layer_i - 1].bias_b.borrow();
    //         let prev_value_a = self.layers[curr_layer_i - 1].value_a.borrow();

    //         for curr_node_i in 0..curr_value_a.len() {

    //             // add the bias
    //             let mut unscaled_z = prev_bias_b[curr_node_i];
        
    //             // add all weights
    //             for prev_node_i in 0..prev_value_a.len() {
    //                 unscaled_z += prev_value_a[prev_node_i] * connection.value_w[(curr_node_i, prev_node_i)];
    //             }
        
    //             // activation function
    //             curr_unscaled_z[curr_node_i] = unscaled_z;
    //             curr_value_a[curr_node_i] = activate_sigmoid(unscaled_z);
    //         }
    //     }
    // }

    fn forward(&mut self, input: &Input) {

        let mut layer0_value_a = self.layers[0].value_a.borrow_mut();

        // layer 0 is input
        assert_eq!(input.len(), layer0_value_a.len(), "Input layers not of same size!");

        // could probably swap layer 1 for input layer?
        for i in 0..input.len() {
            layer0_value_a[i] = input[i]
        }

        drop(layer0_value_a);

        // calculate value for each successive layer
        for curr_layer_i in 1..self.layers.len() {

            let connection = &self.connections[curr_layer_i - 1];

            let curr_layer = &self.layers[curr_layer_i];
            let prev_layer = &self.layers[curr_layer_i - 1];

            let prev_value_a = prev_layer.value_a.borrow();
            let curr_bias_b = curr_layer.bias_b.borrow();

            let mut curr_unscaled_z = curr_layer.unscaled_z.borrow_mut();
            // TODO: assign instead of add

            *curr_unscaled_z = prev_value_a.dot(&connection.value_w.t());
            *curr_unscaled_z += &*curr_bias_b;

            let activated = curr_unscaled_z.map(|unscaled| -> f32 {
                activation_function(*unscaled)
            });

            let mut curr_value_a = curr_layer.value_a.borrow_mut();
            *curr_value_a = activated;
            //self.layers[curr_layer_i].unscaled_z.assign(&(self.layers[curr_layer_i].bias_b.add(&self.layers[curr_layer_i - 1].value_a.dot(&connection.value_w))))
        }
    }

    pub fn get(&mut self, input: &Input) -> Output {
        self.forward(input);

        let layer = &self.layers[self.layers.len() -1];

        let mut out = Vec::with_capacity(layer.len());

        for value in layer.value_a.borrow().iter() {
            out.push(*value)
        };

        out
    }

    // fn backpropagete_old(&mut self, output: &Output) {
    //     let layers_len = self.layers.len();

    //     {
    //         // set last layer error
    //         let last_layer = &mut self.layers[layers_len - 1].borrow_mut();

    //         for i in 0..last_layer.len() {
    //             // TODO: Why nuron.value_a - output[i] and not vice versa?
    //             last_layer.error_z[i] =  (last_layer.value_a[i] - output[i]) * activate_sigmoid_derivate(last_layer.unscaled_z[i])
    //         }
    //     }


    //     // loop for each previous layer
    //     for curr_layer_i in (1..layers_len).rev() {
    //         let mut curr_layer = self.layers[curr_layer_i].borrow_mut();
    //         let mut prev_layer = self.layers[curr_layer_i - 1].borrow_mut();
    //         let connection = self.connections[curr_layer_i - 1].borrow_mut();

    //         for curr_node_i in 0..curr_layer.len() {

    //             for prev_node_i in 0..prev_layer.len() {
    //                 // update connetions

    //                 connection.gradient_w[(curr_node_i, prev_node_i)] += curr_layer.error_z[curr_node_i] * prev_layer.value_a[prev_node_i];
    //             } 

    //             // update bias
    //             curr_layer.gradient_b[curr_node_i] += curr_layer.error_z[curr_node_i];
    //         }

    //         // update previous error
    //         // TODO: Not needed for first layer?
    //         for prev_node_i in 0..prev_layer.len() {
    //             let mut total = 0.0;

    //             for curr_node_i in 0..curr_layer.len() {
    //                 total += curr_layer.error_z[curr_node_i] * connection.value_w[(curr_node_i, prev_node_i)]
    //             }

    //             prev_layer.error_z[prev_node_i] = total * activate_sigmoid_derivate(prev_layer.unscaled_z[prev_node_i])
    //         }
    //     }   
    // }

    fn backpropagete(&mut self, output: &Output) {
        let layers_len = self.layers.len();

        {
            // set last layer error
            let last_layer = &mut self.layers[layers_len - 1].borrow_mut();

            let mut last_layer_error_z = last_layer.error_z.borrow_mut();
            let last_layer_value_a = last_layer.value_a.borrow();
            let last_layer_unscaled_z = last_layer.unscaled_z.borrow();

            last_layer_error_z.iter_mut()
                .zip(last_layer_value_a.iter())
                .zip(output.iter())
                .zip(last_layer_unscaled_z.iter())
                .for_each(|(((error, value), output), unscaled)| {
                    *error = (value - output) * activation_function_derivate(*unscaled)
                });

            // for i in 0..last_layer.len() {
            //     // TODO: Why nuron.value_a - output[i] and not vice versa?
            //     last_layer_error_z[i] =  (last_layer_value_a[i] - output[i]) * activate_sigmoid_derivate(last_layer_unscaled_z[i])
            // }
        }


        // loop for each previous layer
        for curr_layer_i in (1..layers_len).rev() {
            let curr_layer = &self.layers[curr_layer_i];
            let prev_layer = &self.layers[curr_layer_i - 1];
            let connection = &mut self.connections[curr_layer_i - 1].borrow_mut();

            let curr_error_z = curr_layer.error_z.borrow();
            let mut prev_error_z = prev_layer.error_z.borrow_mut();
            let mut curr_gradient_b = curr_layer.gradient_b.borrow_mut();
            let prev_value_a = prev_layer.value_a.borrow();
            let prev_unscaled_z = prev_layer.unscaled_z.borrow();

            for curr_node_i in 0..curr_layer.len() {
                for prev_node_i in 0..prev_layer.len() {
                    // update connetions
                    connection.gradient_w[(curr_node_i, prev_node_i)] += curr_error_z[curr_node_i] * prev_value_a[prev_node_i];
                } 
            }

            // update connetions
            // connection.gradient_w[(curr_node_i, prev_node_i)] += curr_error_z[curr_node_i] * prev_value_a[prev_node_i];
            // connection.gradient_w += curr_error_z.t().dot(&*prev_value_a);
            // let mutli = arr1(xs)
            // connection.gradient_w.axis_iter_mut(Axis(0)).zip(curr_error_z.iter()).for_each(|(mut gradient, error)| {
            //     gradient.add_assign(prev_value_a.(*error))
            // });

            // update bias
            // curr_gradient_b[curr_node_i] += curr_error_z[curr_node_i];
            *curr_gradient_b += &*curr_error_z;


            // update previous error
            *prev_error_z = connection.value_w.t().dot(&*curr_error_z);
            prev_error_z.iter_mut().zip(prev_unscaled_z.iter()).for_each(|(err, unscaled)| {
                *err *= activation_function_derivate(*unscaled);
            })

        }   
    }

    fn clear_gradient(&mut self) {
        for layer in self.layers.iter() {
            for neuron in layer.gradient_b.borrow_mut().iter_mut() {
                *neuron = 0.0;
            }
        }

        // Matrix
        for connections in self.connections.iter_mut() {

            // Current activation
            for connection in connections.gradient_w.iter_mut() {

                // previous activation
                *connection = 0.0;
            }
        }
    }

    fn apply_gradient(&mut self, rate: f32) {
        for layer in self.layers.iter() {
            let mut layer_bias_b = layer.bias_b.borrow_mut();
            let layer_gradient_b = layer.gradient_b.borrow();

            for i in 0..layer.len() {
                layer_bias_b[i] -= layer_gradient_b[i] * rate
            }
        }

        // Matrix
        for connections in self.connections.iter_mut() {

            // Current activation
            // connections.value_w.sub(connections.gradient_w * rate);

            connections.value_w.iter_mut().zip(connections.gradient_w.iter()).for_each(|(value, gradient)| {
                *value -= gradient * rate
            })

            // for i in 0..connections.gradient_w.len() {
            //     // previous activation
            //     for j in 0..connections.gradient_w[i].len() {
            //         connections.value_w[(i, j)] -= connections.gradient_w[(i, j)] * rate;
            //     }
            // }
        }
    }

    pub fn train_samples(&mut self, samples: &[Sample], rate: f32) {
        self.clear_gradient();

        samples.iter().for_each(|sample| {
            self.forward(&sample.input);
            self.backpropagete(&sample.output);
        });
    
        self.apply_gradient(rate / samples.len() as f32);
    }

    fn error(&mut self, sample: &Sample) -> f32 {
        let mut total = 0.0;
    
        self.forward(&sample.input);
    
        let out_layer = &self.layers[self.layers.len() - 1];
    
        assert_eq!(sample.output.len(), out_layer.len(), "Output layers not of same size!");
    
        let out_value_a = out_layer.value_a.borrow();
        for i in 0..sample.output.len() {
            let err = sample.output[i] - out_value_a[i];
            total += err * err
        };
    
        // divide by 2 because the dericative of squaring is 2*(), so they will cancel out
        total / sample.output.len() as f32 / 2.0
    }
    
    pub fn score(&mut self, samples: &Samples) -> f32 {
        let mut total = 0.0;
    
        samples.iter().for_each(|sample| {
            total += self.error(sample)
        });
    
        total / samples.len() as f32
    }
}

enum ActivationFunction {
    Linear(f32),
    Sigmoid
}

// const ACTIVATION_FUNCTION: ActivationFunction = ActivationFunction::Linear(0.0);
// const ACTIVATION_FUNCTION: ActivationFunction = ActivationFunction::Linear(0.01);
const ACTIVATION_FUNCTION: ActivationFunction = ActivationFunction::Sigmoid;

fn activation_function(value: f32) -> f32 {
    match ACTIVATION_FUNCTION {
        ActivationFunction::Sigmoid => {
            1.0 / (1.0 + f32::powf(std::f32::consts::E, -value))
        },
        ActivationFunction::Linear(a) => {
            if value > 0.0 {
                value
            } else {
                value * a
            }
        }
    }
}

fn activation_function_derivate(value: f32) -> f32 {
    match ACTIVATION_FUNCTION {
        ActivationFunction::Sigmoid => {
            let fun = activation_function(value);
            fun * (1.0 - fun)
        },
        ActivationFunction::Linear(a) => {
            if value > 0.0 {
                1.0
            } else {
                a
            }
        }
    }
}