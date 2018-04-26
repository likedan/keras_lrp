import Training as T
import TrainingMemorySafe as TMS

import argparse, sys

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', default="Xception")
parser.add_argument('--train_class_name', default="skirt_length_labels")
parser.add_argument('--training_batch_size', default=64)
parser.add_argument('--learning_rate', default=0.00005)
parser.add_argument('--test_percentage', default=0.05)
parser.add_argument('--memory_safe', default=1)
parser.add_argument('--validation_every_X_batch', default=5)
parser.add_argument('--saving_frequency', default=1)
parser.add_argument('--gpu_num', default=1)
parser.add_argument('--dropout', default=0.2)

args = parser.parse_args()

print(args)

if int(args.memory_safe) == 0:
	trainer = T.Trainer(model_name=args.model_name, train_class_name=args.train_class_name, training_batch_size=int(args.training_batch_size), learning_rate=float(args.learning_rate), test_percentage=float(args.test_percentage), validation_every_X_batch=int(args.validation_every_X_batch))
	trainer.train(steps_per_epoch=64, epochs=5000)
else:
	trainer = TMS.Trainer(model_name=args.model_name, train_class_name=args.train_class_name, gpu_num=int(args.gpu_num), training_batch_size=int(args.training_batch_size), learning_rate=float(args.learning_rate), dropout=float(args.dropout), test_percentage=float(args.test_percentage), validation_every_X_batch=int(args.validation_every_X_batch), saving_frequency=float(args.saving_frequency))
	trainer.train(epochs=5000)
