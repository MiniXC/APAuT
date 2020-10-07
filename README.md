````
Usage: train.py [OPTIONS]

Options:
  --name TEXT                     The name to log to wandb
  --train-sets TEXT               train sets, comma separated if more than one
  --test-set TEXT                 test set
  --model TEXT                    transformers model
  --pad-length INTEGER            length to pad to
  --epochs INTEGER                number of epochs
  --warmup-steps INTEGER          number of warmup steps
  --batch-size INTEGER            batch size
  --gradient-accumulation-steps INTEGER
                                  gradient accumulation steps
  --learning-rate FLOAT
  --weight-decay FLOAT
  --help                          Show this message and exit.
````