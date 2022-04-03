use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::io::BufWriter;
use serde_derive::Deserialize;
use nn::{NN, HaltCondition};

fn main() -> Result<(), Box<dyn Error>> {
    let mut output = File::create("net.json").unwrap();
    let mut writer = BufWriter::new(output);

    let mut reader = csv::Reader::from_path("penguins.csv").expect("Failed to read csv data");
    let mut examples = Vec::with_capacity(400);

    for result in reader.deserialize() {
        // Notice that we need to provide a type hint for automatic
        // deserialization.
        // let record: Record = result?;
        if let Ok(record) = result {
            let record: Record = record;
            println!("{:?}", record);
            if record.species == "Adelie" {
                let expected = if record.sex == Sex::MALE { 1.0 } else { 0.0 };
                examples.push((vec![record.body_mass_g / 10000.0, record.flipper_length_mm / 1000.0], vec![expected]))
            }
        }
    }

    // create examples of the XOR function
    // the network is trained on tuples of vectors where the first vector
    // is the inputs and the second vector is the expected outputs
    /*let examples = [
        (vec![0f64, 0f64], vec![0f64]),
        (vec![0f64, 1f64], vec![1f64]),
        (vec![1f64, 0f64], vec![1f64]),
        (vec![1f64, 1f64], vec![0f64]),
    ];*/

    // create a new neural network by passing a pointer to an array
    // that specifies the number of layers and the number of nodes in each layer
    // in this case we have an input layer with 2 nodes, one hidden layer
    // with 3 nodes and the output layer has 1 node
    let mut net = NN::new(&[2, 4, 1]);

    // train the network on the examples of the XOR function
    // all methods seen here are optional except go() which must be called to begin training
    // see the documentation for the Trainer struct for more info on what each method does
    net.train(&examples)
        .halt_condition(HaltCondition::MSE(15.0))
        .log_interval(Some(100))
        .momentum(0.18)
        .rate(0.5)
        .go();

    let mut success_rate = 0;

    // evaluate the network to see if it learned the XOR function
    for &(ref inputs, ref outputs) in examples.iter() {
        let results = net.run(inputs);
        let (result, key) = (results[0].round(), outputs[0]);
        if result == key {
            success_rate += 1;
        }
        // assert_eq!(result, key);
    }

    println!("Success rate: {} / {}", success_rate, examples.len());

    writeln!(writer, "{}", net.to_json());
    writer.flush().unwrap();

    Ok(())
}

#[derive(Debug, Deserialize)]
struct Record {
    species: String,
    island: String,
    bill_length_mm: f64,
    bill_depth_mm: f64,
    flipper_length_mm: f64,
    body_mass_g: f64,
    sex: Sex
}

#[derive(Debug, Deserialize, PartialEq)]
enum Sex {
    MALE,
    FEMALE
}