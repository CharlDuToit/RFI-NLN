import tensorflow as tf

# model = None
# optimizer = None

_train_step_num = 0


def add_model_regularizer_loss(model):
    # https://stackoverflow.com/questions/62440162/how-do-i-take-l1-and-l2-regularizers-into-account-in-tensorflow-custom-training
    loss=0
    for l in model.layers:
        #if hasattr(l,'layers') and l.layers: # the layer itself is a model
        #    loss+=add_model_loss(l)
        if hasattr(l,'kernel_regularizer') and l.kernel_regularizer and hasattr(l,'kernel') :
            loss += l.kernel_regularizer(l.kernel)
        if hasattr(l,'pointwise_regularizer') and l.pointwise_regularizer and hasattr(l,'pointwise_kernel') :
            loss += l.pointwise_regularizer(l.pointwise_kernel)
        if hasattr(l,'depthwise_regularizer') and l.depthwise_regularizer and hasattr(l,'depthwise_kernel') :
            loss += l.depthwise_regularizer(l.depthwise_kernel)
        if hasattr(l,'bias_regularizer') and l.bias_regularizer:
            loss += l.bias_regularizer(l.bias)
    # print(loss)
    return loss

@tf.function
def train_step(model, x, y, loss_func, optimizer, init_lr=1e-4, epochs=100, epoch=0, lindecay=1.0):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x, training=True)
        # loss = loss_func(x_hat, y)
        func_loss = loss_func(x_hat, y)
        # total_loss = func_loss
        total_loss = func_loss + add_model_regularizer_loss(model)

    # old way with only loss func, no kernel regularization
    # gradients = tape.gradient(loss, model.trainable_variables)

    # new way with kernel regularization
    gradients = tape.gradient(total_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    r = 1 - epoch * (1 - lindecay) / epochs
    lr = init_lr * r
    tf.keras.backend.set_value(optimizer.lr, lr)
    # print(float(tf.keras.backend.get_value(optimizer.learning_rate)))

    # Only return loss function loss, but optimize with total loss
    return func_loss


@tf.function
def train_step_1(model, x, y, loss_func, optimizer, init_lr=1e-4, epochs=100, epoch=0, lindecay=1.0):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x, training=True)
        # loss = loss_func(x_hat, y)
        func_loss = loss_func(x_hat, y)
        # total_loss = func_loss
        total_loss = func_loss + add_model_regularizer_loss(model)

    # old way with only loss func, no kernel regularization
    # gradients = tape.gradient(loss, model.trainable_variables)

    # new way with kernel regularization
    gradients = tape.gradient(total_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    r = 1 - epoch * (1 - lindecay) / epochs
    lr = init_lr * r
    tf.keras.backend.set_value(optimizer.lr, lr)
    # print(float(tf.keras.backend.get_value(optimizer.learning_rate)))

    # Only return loss function loss, but optimize with total loss
    return func_loss


@tf.function
def train_step_2(model, x, y, loss_func, optimizer, init_lr=1e-4, epochs=100, epoch=0, lindecay=1.0):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x, training=True)
        # loss = loss_func(x_hat, y)
        func_loss = loss_func(x_hat, y)
        # total_loss = func_loss
        total_loss = func_loss + add_model_regularizer_loss(model)

    # old way with only loss func, no kernel regularization
    # gradients = tape.gradient(loss, model.trainable_variables)

    # new way with kernel regularization
    gradients = tape.gradient(total_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    r = 1 - epoch * (1 - lindecay) / epochs
    lr = init_lr * r
    tf.keras.backend.set_value(optimizer.lr, lr)
    # print(float(tf.keras.backend.get_value(optimizer.learning_rate)))

    # Only return loss function loss, but optimize with total loss
    return func_loss

@tf.function
def train_step_3(model, x, y, loss_func, optimizer, init_lr=1e-4, epochs=100, epoch=0, lindecay=1.0):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x, training=True)
        # loss = loss_func(x_hat, y)
        func_loss = loss_func(x_hat, y)
        # total_loss = func_loss
        total_loss = func_loss + add_model_regularizer_loss(model)

    # old way with only loss func, no kernel regularization
    # gradients = tape.gradient(loss, model.trainable_variables)

    # new way with kernel regularization
    gradients = tape.gradient(total_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    r = 1 - epoch * (1 - lindecay) / epochs
    lr = init_lr * r
    tf.keras.backend.set_value(optimizer.lr, lr)
    # print(float(tf.keras.backend.get_value(optimizer.learning_rate)))

    # Only return loss function loss, but optimize with total loss
    return func_loss


def get_train_step_func():
    global _train_step_num
    if _train_step_num == 0:
        train_step_func = train_step
    elif _train_step_num == 1:
        train_step_func = train_step_1
    elif _train_step_num == 2:
        train_step_func = train_step_2
    elif _train_step_num == 3:
        train_step_func = train_step_3
    _train_step_num += 1
    return train_step_func


# @tf.function
# def train_step_global(x, y, loss_func):
#     """Executes one training step and returns the loss.
#
#     This function computes the loss and gradients, and uses the latter to
#     update the model's parameters.
#     """
#     with tf.GradientTape() as tape:
#         x_hat = model(x, training=True)
#         loss = loss_func(x_hat, y)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss
#
#
# def train_step(model_, x, y, loss_func, optimizer_):
#     global model
#     global optimizer
#     model = model_
#     optimizer = optimizer_
#     train_step_global(x, y, loss_func)



