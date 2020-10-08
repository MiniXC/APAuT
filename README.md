````
Usage: train.py [OPTIONS]

Options:
  --name TEXT                     The name to log to wandb
  --train-set TEXT                train set
  --lm-set TEXT                   the language model
  --lm-length INTEGER             the language model length
  --lm-epochs INTEGER             epochs to train the lm for, -1 to combine
                                  with train

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
