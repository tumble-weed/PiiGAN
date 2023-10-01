def main():
    config = ng.Config('setting.yml')
    # training data
    with open(config.DATA_FLIST[config.DATASET][0]) as f:
        fnames = f.read().splitlines()
    data = get_train_dataset(fnames,
                             config.IMG_SHAPES, random_crop=config.RANDOM_CROP)
    # images = data.data_pipeline(config.BATCH_SIZE)
    assert False,'TODO_torch dataloader'
    images = Dataloader()
    assert False,'TODO_torch model'
    # main model
    model = PIIGANModel()

    g_vars, d_vars, losses = model.build_graph_with_losses(
        images, config=config)
    
    # validation images
    if config.VAL:
        with open(config.DATA_FLIST[config.DATASET][1]) as f:
            val_fnames = f.read().splitlines()
        # progress monitor by visualizing static images
        for i in range(config.STATIC_VIEW_SIZE):
            static_fnames = val_fnames[i:i+1]
            static_images = ng.data.DataFromFNames(
                static_fnames, config.IMG_SHAPES, nthreads=1,
                random_crop=config.RANDOM_CROP).data_pipeline(1)
            static_inpainted_images = model.build_static_infer_graph(
                static_images, config, name='static_view/%d' % i)
    assert False,'TODO_torch val loader'
    # training settings


    lr = tf.get_variable(
        'lr', shape=[], trainable=False,
        initializer=tf.constant_initializer(1e-4))
    d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
    g_optimizer = d_optimizer
    # gradient processor
    assert False,'TODO_torch optimizer'
    if config.GRADIENT_CLIP:
        gradient_processor = lambda grad_var: (
            tf.clip_by_average_norm(grad_var[0], config.GRADIENT_CLIP_VALUE),
            grad_var[1])
    else:
        gradient_processor = None
    assert False,'TODO_torch move this to train'
    # log dir
    log_prefix = config.MODEL_LOG + '_'.join([
        socket.gethostname(), config.DATASET]) + '_' + str(time.strftime("%m-%d_%H%M"))
    # train discriminator with secondary trainer, should initialize before
    # primary trainer.
    discriminator_training_callback = ng.callbacks.SecondaryTrainer(
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=5,
        graph_def=multigpu_graph_def,
        graph_def_kwargs={
            'model': model, 'data': data, 'config': config, 'loss_type': 'd'},
    )
    assert False,'TODO_torch ?'
    # train generator with primary trainer
    trainer = ng.train.Trainer(
        optimizer=g_optimizer,
        var_list=g_vars,
        max_iters=config.MAX_ITERS,
        graph_def=multigpu_graph_def,
        grads_summary=config.GRADS_SUMMARY,
        gradient_processor=gradient_processor,
        graph_def_kwargs={
            'model': model, 'data': data, 'config': config, 'loss_type': 'g'},
        spe=config.TRAIN_SPE,
        log_dir=log_prefix,
    )
    assert False,'TODO_torch ?'
    # add all callbacks
    if not config.PRETRAIN_COARSE_NETWORK:
        trainer.add_callbacks(discriminator_training_callback)
    assert False,'TODO_torch ?'
    trainer.add_callbacks([
        ng.callbacks.WeightsViewer(),
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix='model_logs/'+config.MODEL_RESTORE+'/snap', optimistic=True),
        ng.callbacks.ModelSaver(config.TRAIN_SPE, trainer.context['saver'], log_prefix+'/snap'),
        ng.callbacks.SummaryWriter((config.VAL_PSTEPS//1), trainer.context['summary_writer'], tf.summary.merge_all()),
    ])
    # launch training
    shutil.copy('./model.py',log_prefix+'/')
    trainer.train()
