import torch 
from torch.optim import Adam,SGD,Adagrad,RMSprop,AdamW


def build_optimizer(cfg,model):



    if cfg.optimizer.lower()=="adam":

        if cfg.freeze==True:
            optimizer=Adam(model.mask_decoder.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        else:
            optimizer=Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    
    elif cfg.optimizer.lower()=="sgd":

        if cfg.freeze==True:
            optimizer=SGD(model.mask_decoder.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        else:
            optimizer=SGD(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    elif cfg.optimizer.lower()=="adagrad":

        if cfg.freeze==True:
            optimizer=Adagrad(model.mask_decoder.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        else:
            optimizer=Adagrad(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    elif cfg.optimizer.lower()=="rmsprop":

        if cfg.freeze==True:
            optimizer=Adagrad(model.mask_decoder.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        else:
            optimizer=Adagrad(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    return optimizer