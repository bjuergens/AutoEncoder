import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gym


class AutoEncoderVAE(nn.Module):

    def __init__(self):
        super(AutoEncoderVAE, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=(2, 2))
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=7, stride=1, padding=(3, 3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)

        # Latent View

        self.lv1 = nn.Linear(5200, 400)
        self.lv2 = nn.Linear(400, 30)
        self.fc_mu = nn.Linear(30, 30)
        self.fc_logsigma = nn.Linear(30, 30)
        self.lv3 = nn.Linear(30, 400)
        self.lv4 = nn.Linear(400, 5200)

        # Decoder
        self.unmaxpool1 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv1 = nn.ConvTranspose2d(40, 20, kernel_size=7, stride=1)
        self.unmaxpool2 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv2 = nn.ConvTranspose2d(20, 3, kernel_size=5, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x, indices1 = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x, indices2 = self.maxpool2(x)
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        features = x.detach().clone()
        x = eps.mul(sigma).add_(mu)
        x = torch.sigmoid(self.lv3(x))
        x = torch.sigmoid(self.lv4(x))
        x = x.view(x.size(0), originalC, originalH, originalW)
        x = self.unmaxpool1(x, indices2)
        x = F.relu(self.deconv1(x))
        x = x[:, :, :52, :40]
        x = self.unmaxpool2(x, indices1, output_size=torch.Size([x.size(0), x.size(1), 210, 160]))
        x = F.relu(self.deconv2(x))
        x = x[:, :, :210, :160]
        return x, features


class FeatureExtractor:
    def __init__(self):
        self.ae = AutoEncoderVAE()
        self.ae.load_state_dict(torch.load('AutoEncoder_RGB_4/Q*Bert_JustQBert/Episode4/ae.pt'))
        self.ae.eval()
        self.last = np.zeros(30)
        self.first = None
        self.count = 0

    def extract(self, obs):
        frame = self.frame2tensor(obs)
        pred, features = self.ae(frame)
        features = features.detach().cpu().numpy().flatten()
        self.count += 1
        if self.count < 10:
            self.first = features
        diff = features - self.first
        # diff *= 10e1
        return diff

    def rencode(self, obs):
        frame = self.frame2tensor(obs)
        pred, features = self.ae(frame)
        pred = pred.permute(0, 2, 3, 1)
        yay = torch.tensor([255], dtype=torch.int)
        imgpred = pred * yay
        return imgpred[0].detach().cpu().numpy().astype(np.uint8)

    @staticmethod
    def normalize(v):
        return v / 255

    @staticmethod
    def frame2tensor(frame):
        frame = FeatureExtractor.normalize(np.array([frame])[:, :, :, :])
        torchFrame = torch.from_numpy(frame).type(torch.FloatTensor)
        torchFrame = torchFrame.permute(0, 3, 1, 2)
        return torchFrame[:, :, :, :]


def feature2img(feature):
    feat_raw = feature.reshape((5, 6))
    feat = (feat_raw * 255).astype(np.uint8)
    feat_data = cv2.cvtColor(feat, cv2.COLOR_GRAY2BGR)
    return cv2.resize(feat_data, (160, 210), interpolation=cv2.INTER_NEAREST)


if __name__ == '__main__':
    fe = FeatureExtractor()

    env = gym.make('QbertNoFrameskip-v4')
    video_obs = cv2.VideoWriter('obs.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (160, 210), True)
    video_pred = cv2.VideoWriter('pred.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (160, 210), True)
    video_feat = cv2.VideoWriter('feat.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (160, 210), True)
    for _ in range(1):
        ob = env.reset()
        done = False
        step = 0
        while not done:
            step += 1
            observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
            video_obs.write(observation.astype(np.uint8))
            video_pred.write(fe.rencode(observation))
            video_feat.write(feature2img(fe.extract(observation)))
            print("step " + str(step))
            if step > 300:
                # done = True
                pass
    env.close()
    cv2.destroyAllWindows()
    video_obs.release()
    video_pred.release()
    video_feat.release()
