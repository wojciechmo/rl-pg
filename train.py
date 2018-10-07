import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import gym, cv2

BATCH_SIZE = 1000
NUM_EPISODES = 10000
LEARNING_RATE = 0.01
GAMMA = 0.99
SAVE_INTERVAL = 500
EVAL_INTERVAL = 100
EPISODE_MAX_ITER = 1000
EPISODE_MIN_REWARD = -500
FC1_NUM, FC2_NUM = 20, 10

def forward(x, num_features, fc1_num, fc2_num, num_actions):

	xavier_init = tf.contrib.layers.xavier_initializer()
	zeros_init = tf.zeros_initializer()

	w1 = tf.Variable(xavier_init([num_features, fc1_num]), name="w1")
	b1 = tf.Variable(zeros_init([fc1_num]), name="b1")
	w2 = tf.Variable(xavier_init([fc1_num, fc2_num]), name="w2")
	b2 = tf.Variable(zeros_init([fc2_num]), name="b2")
	w3 = tf.Variable(xavier_init([fc2_num, num_actions]), name="w3")
	b3 = tf.Variable(zeros_init([num_actions]), name="b3")

	x = tf.nn.relu(tf.matmul(x, w1) + b1)
	x = tf.nn.relu(tf.matmul(x, w2) + b2)
	x = tf.matmul(x, w3) + b3
	
	return x

def compute_returns(rewards, gamma):	
	
	# return - cummulative discounted rewards until the end of episode
	episode_len = len(rewards)
	returns = np.zeros(episode_len)
	return_reward = 0.0
	for i in reversed(range(episode_len)):
		reward = rewards[i]
		return_reward = return_reward * gamma + reward
		returns[i] = return_reward
		
	return returns

def normalize_rewards(rewards):	
	
	mean = np.mean(rewards)
	std = np.std(rewards)
	rewards = rewards - mean
	rewards = rewards/std
	
	return rewards

if __name__ == '__main__':
	
	env = gym.make('LunarLander-v2')
	
	num_features = env.observation_space.shape[0]
	num_actions = env.action_space.n

	states_PH = tf.placeholder(tf.float32, shape=(None, num_features), name="states")
	actions_PH = tf.placeholder(tf.int64, shape=(None,), name="actions")
	returns_PH = tf.placeholder(tf.float32, (None,), name="rewards")

	# agent network forward pass
	y = forward(states_PH, num_features, FC1_NUM, FC2_NUM, num_actions)
	action_probs = tf.nn.softmax(y, name='action_probs')

	# compute loss
	actions_one_hot = tf.one_hot(actions_PH, num_actions) 
	log_y = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=actions_one_hot)
	#log_y = -tf.reduce_sum(tf.log(action_probs) * actions_one_hot) 
	loss = tf.reduce_mean(log_y * returns_PH) # reduce over episode timesteps
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	videowriter = cv2.VideoWriter('./video.mp4', fourcc, 60.0, (600, 400))

	# train for number of episodes
	for episode_iter in range(NUM_EPISODES):

		state = env.reset()
		episode_states, episode_actions, episode_rewards = [], [], []

		num_steps = 0
		total_reward = 0
		
		while True:
			
			if episode_iter % EVAL_INTERVAL == 0:
				img = env.render(mode='rgb_array')		
				img = img[...,::-1]
				videowriter.write(img)

			# pick an action with respect to current agent probs
			probs = sess.run(action_probs, feed_dict = {states_PH: [state]})[0]
			action = np.random.choice(range(num_actions), p=probs)
			next_state, reward, done, _ = env.step(action)

			# store episode samples <s,a,r> for trainig 
			episode_states.append(state)
			episode_actions.append(action)
			episode_rewards.append(reward)
			state = next_state

			total_reward = total_reward + reward	
			if total_reward < EPISODE_MIN_REWARD: done = True		
			if num_steps > EPISODE_MAX_ITER: done = True
			num_steps = num_steps + 1
			
			if done == True:

				episode_states = np.array(episode_states)
				eposode_actions = np.array(episode_actions)
				episode_returns = compute_returns(episode_rewards, GAMMA)
				episode_returns = normalize_rewards(episode_returns)

				# sample batches and train
				batch_iter = 0
				episode_len = len(episode_returns)	
				while batch_iter < episode_len: 
					
					batch_states = episode_states[batch_iter:batch_iter+BATCH_SIZE]
					batch_actions = episode_actions[batch_iter:batch_iter+BATCH_SIZE]
					batch_returns = episode_returns[batch_iter:batch_iter+BATCH_SIZE]
					batch_iter = batch_iter + BATCH_SIZE
					
					# update agent network
					sess.run(train_step, feed_dict={states_PH: batch_states, 
													actions_PH: batch_actions,
													returns_PH: batch_returns})
					
				if (episode_iter + 1) % SAVE_INTERVAL == 0:
					if not os.path.exists('./models'):
						os.makedirs('./models')
					saver.save(sess, './models/net', global_step=episode_iter + 1)

				print "episode:", episode_iter, "reward:", total_reward, 'number of steps:', num_steps
				
				break
	
	videowriter.release()
