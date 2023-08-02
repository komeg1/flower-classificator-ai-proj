import keras
import tensorflow as tf

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x_train, y_train = data

        teacher_predictions = self.teacher(x_train, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x_train, training=True)

            student_loss = self.student_loss_fn(y_train, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            ) * (self.temperature ** 2)
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        trainable_variables = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y_train, student_predictions)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.result()

        results.update({
            "student_loss": student_loss,
            "distillation_loss": distillation_loss
        })
        return results

    def test_step(self, data):
        x_test, y_test = data

        y_prediction = self.student(x_test, training=False)

        student_loss = self.student_loss_fn(y_test, y_prediction)

        self.compiled_metrics.update_state(y_test, y_prediction)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.result()

        results.update({"student_loss": student_loss})
        return results