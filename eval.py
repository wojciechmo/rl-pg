import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import gym, cv2

sess = tf.Session()

saver = tf.train.import_meta_graph('./models/net-2000.meta')
saver.restore(sess, './models/net-2000')
graph = tf.get_default_graph()
states = graph.get_tensor_by_name('states:0')
action_probs = graph.get_tensor_by_name('action_probs:0')

env = gym.envs.make('LunarLander-v2')
num_actions = env.action_space.n

# play episode by sampling over actions from agent policy
state = env.reset()
total_reward, num_steps = 0, 0
done = False

while True:
	
	env.render()
		
	probs = sess.run(action_probs, feed_dict={states: [state]})[0]
	action = np.random.choice(range(num_actions), p=probs)
	#action = np.argmax(probs)
	state, reward, done, _ = env.step(action)

	total_reward = total_reward + reward
	num_steps = num_steps + 1
	
	if done == True:
		break

print 'reward:', total_reward, 'number of steps:', num_steps
