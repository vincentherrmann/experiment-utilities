import torch
import torch.nn.functional as F
import numpy as np
import random
import torchvision
import gym
import json
import base64

from experiment_utilities.trees import tree_map
from experiment_utilities.memory import GeneralMemory
from experiment_utilities.meters import MultiMeter
from experiment_utilities.wandb_logging import Logger
import os
import os.path
import argparse


def get_hyperparameters():
    parser = argparse.ArgumentParser()

    # infrastructure
    parser.add_argument('--wandb_logging', default=0, type=int, required=False)
    parser.add_argument('--use_cpu', default=0, type=int, required=False)
    parser.add_argument('--save_model', default=1, type=int, required=False)
    parser.add_argument('--model_name', default="model.p", type=str, required=False)
    parser.add_argument('--num_workers', default=4, type=int, required=False)
    # general
    parser.add_argument('--seed', default=-1, type=int, required=False)
    # training
    parser.add_argument('--batch_size', default=64, type=int, required=False)
    parser.add_argument('--lr', default=3e-4, type=float, required=False)
    parser.add_argument('--patience', default=10, type=float, required=False)
    parser.add_argument('--num_training_steps', default=100_000, type=float, required=False)
    # environment
    parser.add_argument('--num_envs', default=5, type=int, required=False)
    parser.add_argument('--max_episode_length', default=100, type=int, required=False)
    parser.add_argument('--replay_buffer_size', default=1000, type=int, required=False)
    # model
    parser.add_argument('--num_layers', default=4, type=int, required=False)
    parser.add_argument('--hidden_size', default=128, type=int, required=False)
    # logging
    parser.add_argument('--log_every_n_steps', default=100, type=int, required=False)
    parser.add_argument('--evaluate_every_n_steps', default=1000, type=int, required=False)

    args = parser.parse_args()
    return args


def main(args):
    log_with_wandb = args.wandb_logging > 0
    hyperparameters = dict(vars(args))

    logger = Logger(enabled=log_with_wandb,
                    print_logs_to_console=not log_with_wandb,
                    project="test",
                    tags=["RL"],
                    config=hyperparameters)

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ### ENVIRONMENT ################################################################
    env = gym.vector.make("CartPole-v1", args.num_envs)
    num_observations = env.observation_space.shape[1]
    num_actions = env.action_space[0].n

    ### MODEL ################################################################
    model = torch.nn.Sequential(
        torch.nn.Linear(num_observations + num_actions, args.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_size, num_observations + 2)
    )
    model.to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    class ModelEnv(gym.Env):
        def __init__(self, original_env, model, device='cpu'):
            super().__init__()
            self.observation_space = original_env.observation_space
            self.action_space = original_env.action_space
            self.num_observations = self.observation_space.shape[1]
            self.num_actions = self.action_space[0].n

            self.model = model
            self.current_states = None
            self.device = device

        def reset(self):
            current_states = self.observation_space.sample()
            self.current_states = torch.from_numpy(current_states).to(self.device)
            return current_states

        def step(self, action):
            actions = F.one_hot(torch.from_numpy(action),
                                num_classes=self.num_actions).float().to(self.device)
            model_input = torch.cat([self.current_states,
                                     actions], dim=1)
            model_output = self.model(model_input)
            next_state = model_output[:, :self.num_observations]
            reward = model_output[:, -2]
            done = model_output[:, -1] > 0.5
            self.current_states = next_state
            return next_state.cpu().numpy(), reward.cpu().numpy(), done.cpu().numpy(), []

        def to(self, device):
            self.device = device
            self.current_states.to(device)

    model_env = ModelEnv(original_env=env, model=model, device=device)

    ### POLICY ################################################################
    policy = torch.nn.Sequential(
        torch.nn.Linear(num_observations, args.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_size, num_actions),
        torch.nn.Softmax()
    )
    policy.to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    critic = torch.nn.Sequential(
        torch.nn.Linear(num_actions, args.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_size, 1)
    )
    critic.to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)

    ### MEMORY ################################################################
    memory = GeneralMemory(size=args.replay_buffer_size,
                           artifact_shapes={
                               "observations": (args.max_episode_length+1, env.observation_space.shape[1],),
                               "actions": (args.max_episode_length,),
                               "rewards": (args.max_episode_length,),
                               "done": (args.max_episode_length+1,)},
                           artifact_types={
                               "observations": torch.float,
                               "actions": torch.long,
                               "rewards": torch.float,
                               "done": torch.bool
                           })

    @torch.no_grad()
    def act_in_environment(environment, policy):
        obs = environment.reset()
        obs = torch.from_numpy(obs)

        done = torch.zeros(args.num_envs, dtype=torch.bool)
        obs_rollout = [obs]
        action_rollout = []
        reward_rollout = []
        done_rollout = [done]
        env_step = 0
        while done.sum() < done.numel():
            policy_output = policy(obs.to(device))
            action_dist = torch.distributions.Categorical(policy_output)
            actions = action_dist.sample().cpu().numpy()

            obs, reward, new_done, info = environment.step(actions)
            obs, reward, new_done = tree_map(lambda x: torch.from_numpy(x), (obs, reward, new_done))
            done = torch.logical_or(done, new_done)

            obs_rollout.append(obs)
            action_rollout.append(torch.from_numpy(actions))
            reward_rollout.append(reward)
            done_rollout.append(done)
            env_step += 1

            if env_step >= 200:
                return
        obs_rollout = torch.stack(obs_rollout, dim=1)
        action_rollout = torch.stack(action_rollout, dim=1)
        reward_rollout = torch.stack(reward_rollout, dim=1)
        done_rollout = torch.stack(done_rollout, dim=1)
        return obs_rollout, action_rollout, reward_rollout, done_rollout

    def write_to_memory(obs_rollout, action_rollout, reward_rollout, done_rollout):
        n = args.max_episode_length - action_rollout.shape[1]
        obs_rollout = F.pad(obs_rollout, (0, 0, 0, n), value=0.)
        action_rollout = F.pad(action_rollout, (0, n), value=0)
        reward_rollout = F.pad(reward_rollout, (0, n), value=0)
        done_rollout = F.pad(done_rollout, (0, n), value=True)

        for i in range(obs_rollout.shape[0]):
            memory.store({
                "observations": obs_rollout[i],
                "actions": action_rollout[i],
                "rewards": reward_rollout[i],
                "done": done_rollout[i]})

    ### EVALUATION ################################################################
    @torch.no_grad()
    def evaluate():
        seed = torch.randint(0, 1_000_000, size=(1,)).item()
        torch.manual_seed(123)
        model.eval()

        total_num_examples = 0
        total_loss = 0
        total_num_correct = 0
        for eval_step, batch in enumerate(eval_dataloader):
            if eval_step >= 10000:
                break
            batch = tree_map(lambda x: x.to(model_engine.local_rank), batch)
            x, target = batch
            if fp16:
                x = x.half()
            batch_size = x.shape[0]

            model_output = model(x.view(batch_size, 784))

            losses = F.cross_entropy(model_output, target, reduce='none')

            predictions = model_output.argmax(dim=1)
            correct_predictions = predictions == target
            total_num_correct += correct_predictions.sum().item()
            total_loss += losses.sum().item()
            total_num_examples += batch_size
        eval_loss = total_loss / total_num_examples
        eval_accuracy = total_num_correct / total_num_examples
        torch.manual_seed(seed)
        model.train()
        return eval_loss, eval_accuracy

    meter = MultiMeter()
    total_environment_steps = 0
    iteration = 0
    epoch = 0
    lowest_eval_loss = float('inf')
    patience_count = 0

    def update_model(num_training_steps):
        for step in range(num_training_steps):
            batch = memory.sample_batch(args.batch_size)
            batch = tree_map(lambda x: x.to(device), batch)
            actions = F.one_hot(batch["actions"], num_classes=num_actions).float()
            model_input = torch.cat([batch["observations"][:, :-1],
                                     actions], dim=2).view(-1, num_observations + num_actions)
            target = torch.cat([batch["observations"][:, 1:],
                                batch["rewards"].unsqueeze(2).float(),
                                batch["done"][:, 1:].unsqueeze(2).float()], dim=2).view(-1, num_observations + 2)

            model_output = model(model_input)
            losses = F.mse_loss(model_output, target, reduction='none').view(args.batch_size, -1, num_observations + 2)
            masked_losses = losses * torch.logical_not(batch["done"][:, :-1]).unsqueeze(2)
            loss = masked_losses.mean()

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

    def update_policy(num_training_steps):
        return


    ### MAIN LOOP ################################################################
    while iteration < args.num_training_steps:
        # act
        rollouts = act_in_environment(environment=env, policy=policy)
        write_to_memory(*rollouts)

        # update model
        update_model(num_training_steps=100)

        virtual_rollouts = act_in_environment(environment=model_env, policy=policy)

        # update policy


        # update policy
        for batch in train_dataloader:
            batch = tree_map(lambda x: x.to(model_engine.local_rank), batch)
            x, target = batch
            if fp16:
                x = x.half()
            batch_size = x.shape[0]

            model_output = model(x.view(batch_size, 784))

            loss = F.cross_entropy(model_output, target)
            model_engine.backward(loss)
            model_engine.step()

            meter.update({
                "train loss": loss.item()
            })

            if iteration % args.log_every_n_steps == 0:
                logger().log({
                    "train loss": meter["train loss"].avg,
                    "epoch": epoch,
                }, step=iteration)
                meter.reset()

            if iteration % args.evaluate_every_n_steps == 0:
                eval_loss, eval_acc = evaluate()
                logger().log({
                    "eval loss": eval_loss,
                    "eval accuracy": eval_acc,
                }, step=iteration)

                if eval_loss < lowest_eval_loss:
                    lowest_eval_loss = eval_loss
                    patience_count = 0
                    if args.save_model and log_with_wandb and model_engine.local_rank == 0:
                        torch.save(
                            model.cpu(),
                            os.path.join(logger().run.dir, args.model_name),
                        )
                        model.to(model_engine.local_rank)
                else:
                    patience_count += 1
                    if patience_count >= args.patience:
                        return
            iteration += 1
        epoch += 1


if __name__ == '__main__':
    args = get_hyperparameters()
    main(args)