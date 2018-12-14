
import copy
import os
from collections import deque
from math import ceil, exp

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage.filters import gaussian_filter  # 画像をぼやかす

import torchvision.transforms as T

''' オプション '''
EPISODE_MAX = 500
MASK_SIGMA = 5
BLUR_SIGMA = 3
STRIDE = 4  # SaliencyMapの粒度(小さいほど高精度)
BATCH_SIZE = 16
resdir = 'cartpole/'
''''''

steps_done = 0
# CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.shape[0], -1))
        # x = F.sigmoid(x)
        return x


def get_screen(env):
    ''' スクリーン画像 '''
    # (色, 縦, 横) = (3, 800, 1200)
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # 不要部分削除 -> (3, 320, 720)
    h, w = screen.shape[1:]
    screen = screen[:, int(h * 2 / 5):int(h * 4 / 5),
                    int(w * 1 / 5): int(w * 4 / 5)]

    # 加工
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])
    screen = screen.astype(np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = resize(screen).to(device)  # (3, 40, 90)
    return screen


def select_action(env, model, state, eps_start=0.9, eps_end=0.05, eps_decay=200):
    ''' 行動決定 '''
    global steps_done
    eps_threshold = eps_end + (eps_start - eps_end) * \
        exp(-steps_done / eps_decay)
    steps_done += 1
    if np.random.rand() > eps_threshold:
        with torch.no_grad():
            return model(state).argmax().item()
    else:
        return env.action_space.sample()


def make_mask(size, point):
    ''' マスク画像 '''
    x, y = np.ogrid[-point[0]:size[0] - point[0],
                    -point[1]:size[1] - point[1]]
    keep = x * x + y * y <= 1  # pointの近くだけTrue
    mask = np.zeros(size)
    mask[keep] = 1
    mask = gaussian_filter(mask, sigma=MASK_SIGMA)
    mask = mask / mask.max()  # 正規化
    return mask


def make_blurred_image(image, point):
    ''' マスクとの合成画像 '''
    image = image.cpu().numpy() * 1
    mask = make_mask(image.shape[1:], point)
    for i in range(3):
        blurred_frame = gaussian_filter(image[i], sigma=BLUR_SIGMA)
        image[i] = (image[i] * (1 - mask)) + (blurred_frame * mask)
    image = torch.from_numpy(image).to(device)
    return image


def make_saliency_map(model, image):
    ''' Saliency Map '''
    model.eval()  # 評価モード
    torch.no_grad()  # 履歴保存なし

    hight, width = image.shape[1:]

    # 通常の出力
    normal_q = model(image)

    # Map初期化
    saliency_map = np.zeros(
        (3, ceil(hight / STRIDE), ceil(width / STRIDE)))

    # Map作成
    for h in range(STRIDE // 2, hight, STRIDE):
        for w in range(STRIDE // 2, width, STRIDE):
            # ぼかし画像の出力
            blurred_image = make_blurred_image(image, [h, w])
            perturbed_q = model(blurred_image)

            # Saliency Scoreを色値に設定
            saliency_score = (normal_q - perturbed_q).pow(2).sum() / 2
            saliency_map[:, h // STRIDE, w // STRIDE] = \
                np.array([saliency_score, saliency_score, saliency_score])

    # 正規化
    saliency_map /= saliency_map.max()

    return saliency_map


def optimize_model(model, optimizer, memory, gamma=0.999):
    ''' 学習 '''
    if len(memory) < BATCH_SIZE:
        return

    # バッチ
    batch = np.random.choice(memory, BATCH_SIZE)
    state_batch = torch.stack([b['state'] for b in batch]).to(device)
    action_batch = torch.tensor([b['action'] for b in batch]).to(device)
    reward_batch = torch.tensor([b['reward'] for b in batch]).to(device)

    # next_stateはNoneが混ざってるから別処理
    next_state = [b['next_state'] for b in batch]
    non_final_mask = torch.tensor([s is not None for s in next_state],
                                  device=device, dtype=torch.uint8)  # [BATCH_SIZE or so]
    non_final_next_states = torch.stack(
        [s for s in next_state if s is not None]).to(device)  # [BATCH_SIZE or so, 3, 40, 90]

    # 予測値
    state_action_values = model(state_batch)  # [BATCH_SIZE, 2]
    state_action_values = state_action_values.gather(
        1, action_batch.unsqueeze(1))  # 最大値だけ残す -> [BATCH_SIZE, 1]
    # print(state_action_values, type(state_action_values), state_action_values.shape)

    # 期待値
    next_state_values = torch.zeros(BATCH_SIZE, device=device)  # [BATCH_SIZE]
    with torch.no_grad():
        next_state_values[non_final_mask] = model(
            non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # パラメータ更新
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_movie(image_sequence, savename):
    ''' GIF保存 '''
    movie = []
    for image in image_sequence:
        image = Image.fromarray(
            np.uint8(image * 255).transpose(1, 2, 0))
        image = image.resize((720, 320))
        movie.append(image)
    movie[0].save(savename + '.gif', optimize=False, save_all=True,
                  append_images=movie[1:], duration=150, loop=0)


def save_blended_movie(image_seq1, image_seq2, savename):
    ''' 合成GIF保存 '''
    movie = []
    for image1, image2 in zip(image_seq1, image_seq2):
        image1 = Image.fromarray(
            np.uint8(image1 * 255).transpose(1, 2, 0))
        image2 = Image.fromarray(
            np.uint8(image2 * 255).transpose(1, 2, 0))
        image1 = image1.resize((720, 320))
        image2 = image2.resize((720, 320))
        image = Image.blend(image1, image2, 0.5)
        movie.append(image)
    movie[0].save(savename + '.gif', optimize=False, save_all=True,
                  append_images=movie[1:], duration=150, loop=0)


def main():
    os.makedirs(resdir, exist_ok=True)

    # モデル
    model = DQN().to(device)
    optimizer = torch.optim.RMSprop(model.parameters())
    # print(model)

    # ゲーム作成
    env = gym.make('CartPole-v0').unwrapped

    ''' 学習 '''
    memory = deque(maxlen=200)
    max_t = 0
    for i_episode in range(EPISODE_MAX):
        print(f'エピソード {i_episode} : ', end='')
        env.reset()

        # 初期状態
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen

        for t in range(int(1e10)):
            # 行動
            action = select_action(env, model, state)
            _, reward, done, _ = env.step(action)
            # print('%d' % reward, end='')

            # 遷移先
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen
                optratio = 0.3
            else:
                next_state = None
                reward = 0
                optratio = 1

            if np.random.rand() < optratio:
                # データ保存
                memory.append({'state': state, 'action': action,
                               'reward': reward, 'next_state': next_state})
                # 学習
                optimize_model(model, optimizer, memory)

            # 状態遷移
            state = next_state

            # 終了判定
            if done:
                break
        max_t = max(max_t, t)
        print(f'試行回数 {t} 回 / 最大 {max_t} 回')

    # モデル保存
    torch.save(model.state_dict(), f'{resdir}model.pth')
    # model.load_state_dict(torch.load(f'{resdir}model.pth'))

    # print('Saving result ...')
    # screen_hist, state_hist, saliency_map_hist = [], [], []
    # for e in range(3):
    #     env.reset()
    #     # 初期状態
    #     last_screen = get_screen(env)
    #     for t in range(int(1e10)):
    #         # 状態遷移
    #         last_screen = current_screen
    #         current_screen = get_screen(env)
    #         state = current_screen - last_screen
    #
    #         # Saliency Map
    #         saliency_map = make_saliency_map(model, state)
    #
    #         # 履歴追加
    #         screen_hist.append(current_screen)
    #         state_hist.append(state)
    #         saliency_map_hist.append(saliency_map)
    #
    #         # 行動
    #         action = select_action(env, model, state)
    #         _, reward, done, _ = env.step(action)
    #
    #         # 終了判定
    #         if done:
    #             break
    #     print('\n試行回数', t, '回')
    #
    # # GIF保存
    # save_movie(screen_hist,  f'{resdir}screen')
    # save_movie(state_hist,  f'{resdir}state')
    # save_movie(saliency_map_hist,  f'{resdir}saliency_map')
    # save_blended_movie(screen_hist, saliency_map_hist,  f'{resdir}blended')

    # ゲーム終了
    env.close()


if __name__ == '__main__':
    main()
