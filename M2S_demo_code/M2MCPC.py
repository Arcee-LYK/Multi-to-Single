import torch
import torch.nn as nn

class M2M_CPC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_step=1):
        super(M2M_CPC, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim #hidden_dim == context_dim
        self.num_layers = num_layers
        self.n_prediction_steps = n_prediction_steps
        self.min_start_step = min_start_step
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        self.eeg_ar_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=context_dim, num_layers=num_layers, batch_first=True)
        self.eye_ar_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=context_dim, num_layers=num_layers, batch_first=True)

        self.eeg_predictors = nn.ModuleList([nn.Linear(context_dim * 2, embedding_dim) for _ in range(n_prediction_steps)])
        self.eye_predictors = nn.ModuleList([nn.Linear(context_dim * 2, embedding_dim) for _ in range(n_prediction_steps)])

    def forward(self, eeg_vq, eye_vq):
        batch_size, time_length, _ = eeg_vq.size()
        t_samples = (torch.randint(time_length - self.n_prediction_steps - self.min_start_step, size=(1,)) + self.min_start_step).long()
        nce = 0
        eeg_encode_samples = torch.empty((self.n_prediction_steps, batch_size, self.embedding_dim), device=eeg_vq.device).double()
        eye_encode_samples = torch.empty((self.n_prediction_steps, batch_size, self.embedding_dim), device=eeg_vq.device).double()
        # print("t_samples:", t_samples)
        for i in range(1, self.n_prediction_steps+1):
            eeg_encode_samples[i-1] = eeg_vq[:, t_samples+i, :].reshape(batch_size, self.embedding_dim)
            eye_encode_samples[i-1] = eye_vq[:, t_samples+i, :].reshape(batch_size, self.embedding_dim)
        eeg_forward_seq = eeg_vq[:, :t_samples+1, :]
        eye_forward_seq = eye_vq[:, :t_samples+1, :]
        # print("eeg_encode_samples:", eeg_encode_samples.shape)
        # print("eeg_forward_seq:", eeg_forward_seq.shape)

        eeg_hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=eeg_vq.device).float(),
                      torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=eeg_vq.device).float())
        eeg_context, eeg_hidden = self.eeg_ar_lstm(eeg_forward_seq, eeg_hidden)
        # print("eeg_context:", eeg_context.shape) #[batch_size, 4, 64]
        # print("eeg_hidden:", eeg_hidden[0].shape)

        eye_hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=eye_vq.device).float(),
                      torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=eye_vq.device).float())
        eye_context, eye_hidden = self.eye_ar_lstm(eye_forward_seq, eye_hidden)

        eeg_context = eeg_context[:, t_samples, :].reshape(batch_size, self.context_dim)
        eye_context = eye_context[:, t_samples, :].reshape(batch_size, self.context_dim)
        # print("eeg_context:", eeg_context.shape) #[batch_size, 64]

        eeg_pred = torch.empty((self.n_prediction_steps, batch_size, self.embedding_dim), device=eeg_vq.device).double()
        eye_pred = torch.empty((self.n_prediction_steps, batch_size, self.embedding_dim), device=eye_vq.device).double()
        for i in range(0, self.n_prediction_steps):
            eeg_linear = self.eeg_predictors[i]
            eeg_inp = torch.cat((eeg_context, eye_context), dim=1)
            eeg_pred[i] = eeg_linear(eeg_inp)
            eye_linear = self.eye_predictors[i]
            eye_inp = torch.cat((eye_context, eeg_context), dim=1)
            eye_pred[i] = eye_linear(eye_inp)

        for i in range(0, self.n_prediction_steps):
            total1 = torch.mm(eeg_encode_samples[i], torch.transpose(eye_pred[i], 0, 1))
            total2 = torch.mm(eye_encode_samples[i], torch.transpose(eeg_pred[i], 0, 1))
            total3 = torch.mm(eeg_encode_samples[i], torch.transpose(eeg_pred[i], 0, 1))
            total4 = torch.mm(eye_encode_samples[i], torch.transpose(eye_pred[i], 0, 1))

            nce +=  torch.sum(torch.diag(self.lsoftmax(total1)))
            nce +=  torch.sum(torch.diag(self.lsoftmax(total2)))
            nce +=  0.1 * torch.sum(torch.diag(self.lsoftmax(total3)))
            nce +=  0.1 * torch.sum(torch.diag(self.lsoftmax(total4)))

        nce /= -1. * batch_size * self.n_prediction_steps
        return  nce

