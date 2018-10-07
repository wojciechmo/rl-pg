# Policy Gradients

Teach an agent how to play Lunar Lander with Policy Gradients and TensorFlow.

<img src="https://github.com/WojciechMormul/rl-pg/blob/master/imgs/games.png" width="1200">

Policy-based methods avoid learning value function and instrad directly find optimal agent's policy. Simple cross entropy method depends on playing some games with current policy, finding elite games which have rewrad better than others, and directly changing policy based on states and actions in those elite games. Policy gradients method is a bit more sophisticated.

Find optimal policy parameters which maximize return - cummulative discounted rewards:

<img src="https://github.com/WojciechMormul/rl-pg/blob/master/imgs/img1.png" width="240">

Gradient of objective function with respect to policy parameters theta:

<img src="https://github.com/WojciechMormul/rl-pg/blob/master/imgs/img2.png" width="540">

Gradient can be estimated with Monte Carlo Sampling:

<img src="https://github.com/WojciechMormul/rl-pg/blob/master/imgs/img3.png" width="400">

For policy parameters update take only samples from single episode. Loss can be presented as follows:

<img src="https://github.com/WojciechMormul/rl-pg/blob/master/imgs/img4.png" width="360">

Minimize loss with gradient descent to find optimal policy.

<img src="https://github.com/WojciechMormul/rl-pg/blob/master/imgs/img5.png" width="200">

Do it for each action in episode.

<img src="https://github.com/WojciechMormul/rl-pg/blob/master/imgs/img6.png" width="350">

During trainig sample action with respect to probability distibution returned by current policy. Reward shaping is necassaray or at least very helpful. Moreover trainig data could be decorellated by taking trainig batches from many different episodes but it's optional.

