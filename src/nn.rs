use rand::Rng;

#[derive(Debug)]
struct Neuron {
    value_a: f32,
    unscaled_z: f32,
    error_z: f32,
    bias_b: f32,
    gradient_b: f32
}

#[derive(Debug)]
struct Connection {
    value_w: f32,
    gradient_w: f32,
}

type Layer = Vec<Neuron>;
pub type Input = Vec<f32>;
pub type Output = Vec<f32>;
pub struct Sample {
    pub input: Input,
    pub output: Output,
}

pub type Samples = Vec<Sample>;


fn new_layer(size: usize) -> Layer {
    let mut rng = rand::thread_rng();
    let mut neurons = Vec::with_capacity(size);

    // Init each neuron with a random bias
    for _ in 0..size as usize {
        neurons.push(Neuron {
            value_a: 0.0,
            unscaled_z: 0.0,
            error_z: 0.0,
            bias_b: rng.gen_range(-1.0..1.0),
            gradient_b: 0.0
        })
    }

    neurons
}

type Connections = Vec<Vec<Connection>>;

fn new_connections(current: &Layer, previous: &Layer) -> Connections {
    let mut rng = rand::thread_rng();
    let mut connections = Vec::with_capacity(current.len());

    // Init each connection with a random weight
    for _ in 0..current.len() {
        let mut current_connections = Vec::with_capacity(previous.len());

        for _ in 0..previous.len() {                
            current_connections.push(Connection {
                value_w: rng.gen_range(-1.0..1.0),
                gradient_w: 0.0,
            })
        }

        connections.push(current_connections)
    }

    connections
}

#[derive(Debug)]
pub struct NN {
    layers: Vec<Layer>,
    connections: Vec<Connections>
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
    fn forward(&mut self, input: &Input) {

        // layer 0 is input
        assert_eq!(input.len(), self.layers[0].len(), "Input layers not of same size!");

        // could probably swap layer 1 for input layer?
        for i in 0..input.len() {
            self.layers[0][i].value_a = input[i]
        }

        // calculate value for each successive layer
        for curr_layer_i in 1..self.layers.len() {

            let connection = &self.connections[curr_layer_i - 1];

            let layers_len = self.layers[curr_layer_i].len();
            for curr_node_i in 0..layers_len {
                let prev_layer = &self.layers[curr_layer_i - 1];

                let curr_connection = &connection[curr_node_i];
                let curr_node = &self.layers[curr_layer_i][curr_node_i];

                // add the bias
                let mut unscaled_z = curr_node.bias_b;
        
                // add all weights
                for prev_node_i in 0..prev_layer.len() {
                    unscaled_z += prev_layer[prev_node_i].value_a * curr_connection[prev_node_i].value_w;
                }
        
                // activation function
                self.layers[curr_layer_i][curr_node_i].unscaled_z = unscaled_z;
                self.layers[curr_layer_i][curr_node_i].value_a = activate_sigmoid(unscaled_z);
            }
        }
    }

    fn backpropagete(&mut self, output: &Output) {
        // set last layer error
        let layers_len = self.layers.len();
        let last_layer = &mut self.layers[layers_len - 1];

        for i in 0..last_layer.len() {
            let neuron = &mut last_layer[i];

            neuron.error_z =  (neuron.value_a - output[i]) * activate_sigmoid_derivate(neuron.unscaled_z)
        }

        // loop for each previous layer
        for curr_layer_i in (1..layers_len).rev() {
            for curr_node_i in 0..self.layers[curr_layer_i].len() {

                for prev_node_i in 0..self.layers[curr_layer_i - 1].len() {
                    // update connetions

                    let prev_node = &self.layers[curr_layer_i - 1][prev_node_i];

                    let connection = &mut self.connections[curr_layer_i - 1][curr_node_i][prev_node_i];
                    connection.gradient_w += self.layers[curr_layer_i][curr_node_i].error_z * prev_node.value_a;
                } 

                // update bias
                self.layers[curr_layer_i][curr_node_i].gradient_b += self.layers[curr_layer_i][curr_node_i].error_z;
            }

            // update previous error
            // TODO: Not needed for first layer?
            for prev_node_i in 0..self.layers[curr_layer_i - 1].len() {
                let mut total = 0.0;

                for curr_node_i in 0..self.layers[curr_layer_i].len() {
                    total += self.layers[curr_layer_i][curr_node_i].error_z * self.connections[curr_layer_i - 1][curr_node_i][prev_node_i].value_w
                }

                let prev_node = &mut self.layers[curr_layer_i - 1][prev_node_i];
                prev_node.error_z = total * activate_sigmoid_derivate(prev_node.unscaled_z)
            }
        }   
    }

    fn clear_gradient(&mut self) {
        for layer in self.layers.iter_mut() {
            for neuron in layer.iter_mut() {
                neuron.gradient_b = 0.0;
            }
        }

        // Matrix
        for connections in self.connections.iter_mut() {

            // Current activation
            for connections in connections.iter_mut() {

                // previous activation
                for connection in connections.iter_mut() {
                    connection.gradient_w = 0.0;
                }
            }
        }
    }

    fn apply_gradient(&mut self, rate: f32) {
        for layer in self.layers.iter_mut() {
            for neuron in layer.iter_mut() {
                neuron.bias_b -= neuron.gradient_b * rate;
            }
        }

        // Matrix
        for connections in self.connections.iter_mut() {

            // Current activation
            for connections in connections.iter_mut() {

                // previous activation
                for connection in connections.iter_mut() {
                    connection.value_w -= connection.gradient_w * rate;
                }
            }
        }
    }

    pub fn train_samples(&mut self, samples: &Samples, rate: f32) {
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
    
    
        for i in 0..sample.output.len() {
            let err = sample.output[i] - out_layer[i].value_a;
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

fn activate_linear(value: f32) -> f32 {
    if value > 0.0 {
        value
    } else {
        0.0
    }
}

fn activate_sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + f32::powf(std::f32::consts::E, -value))
}

fn activate_sigmoid_derivate(value: f32) -> f32 {
    let fun = activate_sigmoid(value);
    fun * (1.0 - fun)
}