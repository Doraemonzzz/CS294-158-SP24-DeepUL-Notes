

## ELBO

我们的目标是最大化对数似然：
$$
\mathbb E_{x\sim p} \log p(x)
$$
所以核心问题是计算：
$$
\log p(x)
$$
当上式优化好之后，我们通过$p(.)$采样$x$。

但一个问题是，上式往往很难计算，我们假设有隐变量$z$，那么：
$$
\log p(x)= \log \int  p(x, z) dz
$$
考虑任意分布$q(z)$，那么：
$$
\log p(x)= \log \int  p(x, z) dz=\log \int  q(z)\frac{p(x, z)}{q(z)} dz
=\log \mathbb E_{z\sim q(.)}\left[ \frac{p(x, z)}{q(z)} \right]
\ge \mathbb E_{z\sim q(.)}\log \left[ \frac{p(x, z)}{q(z)} \right]
$$
另一方面：
$$
\begin{aligned}
\log p(x)
& = \mathbb E_{z\sim q(.)} \left [\log p(x) \right] \\
& =\mathbb E_{z\sim q(.)} \left [\log\frac{ p(x, z)}{p(z|x)} \right]  \\
&= \mathbb E_{z\sim q(.)} \left [\log\frac{ p(x, z)}{q(z)}\frac{q(z)}{p(z|x)} \right]  \\
&= \mathbb E_{z\sim q(.)} \left [\log\frac{ p(x, z)}{q(z)} \right]+ \mathbb E_{z\sim q(.)}\left[\frac{q(z)}{p(z|x)} \right] \\
&= \mathbb E_{z\sim q(.)} \left [\log\frac{ p(x, z)}{q(z)} \right]+ D_{\mathrm{KL}}(q(z)\ \|\  p(z|x)) \\
& \ge \mathbb E_{z\sim q(.)} \left [\log\frac{ p(x, z)}{q(z)} \right]

\end{aligned}
$$
进一步变换可得：
$$
\begin{aligned}
\mathbb E_{z\sim q(.)} \left [\log\frac{ p(x, z)}{q(z)} \right]
&= \mathbb E_{z\sim q(.)} \left [\log\frac{ p(x| z) \times p(z)}{q(z)} \right]  \\
&=  \mathbb E_{z\sim q(.)} \left [\log { p(x| z) } \right]
+ \mathbb E_{z\sim q(.)} \left [\log\frac{  p(z)}{q(z)} \right] \\
&=  \mathbb E_{z\sim q(.)} \left [\log { p(x| z) } \right] -
 \mathbb E_{z\sim q(.)} \left [\log\frac{  q(z)}{p(z)} \right]  \\
& =\mathbb E_{z\sim q(.)} \left [\log{ p(x| z) } \right] -  D_{\mathrm{KL}}(q(z)\ \|\  p(z))
\end{aligned}
$$
等效的，也等于：
$$
\begin{aligned}
\mathbb E_{z\sim q(.)} \left [\log\frac{ p(x, z)}{q(z)} \right]
&= \mathbb E_{z\sim q(.)} \left [\log\frac{ p(z) \times p(x| z) }{q(z)} \right]  \\
&=  \mathbb E_{z\sim q(.)} \left [\log { p(z) } \right]
+ \mathbb E_{z\sim q(.)} \left [\log\frac{  p(x|z)}{q(z)} \right] \\
&=  \mathbb E_{z\sim q(.)} \left [\log { p(z) } \right] -
 \mathbb E_{z\sim q(.)} \left [\log\frac{  q(z)}{p(x|z)} \right]  \\
& =\mathbb E_{z\sim q(.)} \left [\log{ p(z) } \right] -  D_{\mathrm{KL}}(q(z)\ \|\  p(x|z))
\end{aligned}
$$



### 特殊情况

如果取$q(z)=q(z|x)$，那么上述公式变成：
$$
\begin{aligned}
\log p(x)
&= \log \mathbb E_{z\sim q(.|x)}\left[ \frac{p(x, z)}{q(z|x)} \right] \\
& \ge \mathbb E_{z\sim q(.|x)} \left [\log\frac{ p(x, z)}{q(z|x)} \right]  \\
\mathbb E_{z\sim q(.|x)} \left [\log\frac{ p(x, z)}{q(z|x)} \right]& =
\mathbb E_{z\sim q(.|x)} \left [\log{ p(x| z) } \right] -  D_{\mathrm{KL}}(q(z|x)\ \|\  p(z))  \\
&= \mathbb E_{z\sim q(.|x)} \left [\log{ p(z) } \right] -  D_{\mathrm{KL}}(q(z|x)\ \|\  p(x|z))
\end{aligned}
$$



### 为什么选择$q(z)=q(z|x)$

假设我们的目标是最小化：
$$
D_{\mathrm{KL}}(q(x, z)\ \|\  p(x, z))
=\iint  q(x,z)\log \frac{q(x, z)}{p(x, z)} dxdz
$$
注意到：
$$
\begin{aligned}
\int q(x,z)\log \frac{q(x, z)}{p(x, z)} dx
&= \iint q(z|x)q(x)\log \frac{q(z|x)q(x)}{p(x, z)} dxdz \\
&= \int q(x) \left [\int q(z|x )\left( \log q(z|x)+\log q(x)-\log p(x, z)\right ) dz \right] dx \\
&= \int q(x) \left [\int q(z|x )\log q(z|x) dz -\int q(z|x )\log p(x, z) dz +\log q(x)\right] dx \\
&= \int q(x) \log q(x) dx + \int q(x)\int  q(z|x )\log\frac{ q(z|x)}{ p(x, z)} dz dx
\end{aligned}
$$
注意我们的目标是找到$q$，使得$D_{\mathrm{KL}}(q(x, z)\ \|\  p(x, z)) $最小化，所以最小化上式第二项即可：
$$
\mathcal L =-\mathbb E_{x\sim q(.)}
\left [
\mathbb E_{z\sim q(.|x) }\frac{p(x, z)}{q(z|x)}
\right]
$$
大括号内部即特殊情况下的式子。



## ELBO推广

现在将$p(x)$替换为$p(x|y)$，$q(z)$替换为$q(z|x, y)$，那么：
$$
\log p(x| y)= \log \int  \frac{p(x,y)}{p(y)} dz=\log \int  q(z)\frac{p(x,y, z)}{q(z)p(y)} dz
=\log \mathbb E_{z\sim q(.)}\left[ \frac{p(x,y, z)}{q(z)p(y)} \right]
\ge \mathbb E_{z\sim q(.)}\log \left[  \frac{p(x,y, z)}{q(z)p(y)} \right]
$$
另一方面：
$$
\begin{aligned}
\log p(x|y)
& = \mathbb E_{z\sim q(.)} \left [\log p(x|y) \right] \\
& = \mathbb E_{z\sim q(.)} \left [\log \frac{p(x,y)}{p(y)} \right] \\
& =\mathbb E_{z\sim q(.)} \left [\log\frac{ p(x, y, z)}{p(y)p(z|x,y)} \right]  \\
&= \mathbb E_{z\sim q(.)} \left [\log\frac{ p(x,y, z)}{p(y)q(z)}\frac{q(z)}{p(z|x, y)} \right]  \\
&= \mathbb E_{z\sim q(.)} \left [\log\frac{ p(x, y, z)}{p(y)q(z)} \right]+ \mathbb E_{z\sim q(.)}\left[\frac{q(z)}{p(z|x,y)} \right] \\
&= \mathbb E_{z\sim q(.)} \left [\log\frac{ p(x, y, z)}{p(y)q(z)} \right]+ D_{\mathrm{KL}}(q(z)\ \|\  p(z|x,y)) \\
& \ge \mathbb E_{z\sim q(.)} \left [\log\frac{ p(x,y, z)}{p(y)q(z)} \right]

\end{aligned}
$$
进一步变换可得：
$$
\begin{aligned}
 \mathbb E_{z\sim q(.)}  \left [\log\frac{ p(x,y, z)}{p(y)q(z)} \right]
&= \mathbb E_{z\sim q(.)}  \left [\log\frac{ p(x,y, z)p(y,z)}{p(y,z)p(y)q(z)} \right]  \\
&=  \mathbb E_{z\sim q(.)} \left [\log { p(x| y,z) } \right]
+ \mathbb E_{z\sim q(.)} \left [\log\frac{  p(z|y)}{q(z)} \right] \\
&=  \mathbb E_{z\sim q(.)} \left [\log { p(x| y,z) } \right] -
 \mathbb E_{z\sim q(.)} \left [\log\frac{  q(z)}{p(z|y)} \right]  \\
& =\mathbb E_{z\sim q(.)} \left [\log{ p(x| y,z) } \right] -  D_{\mathrm{KL}}(q(z)\ \|\  p(z|y))
\end{aligned}
$$


### 特殊情况

如果取$q(z)=q(z|x,y)$，那么上述公式变成：
$$
\begin{aligned}
\log p(x|y)
&= \log \mathbb E_{z\sim q(.|x,y)}\left[ \frac{p(x, y, z)}{p(y)q(z|x, y)} \right] \\
& \ge \mathbb E_{z\sim q(.|x,y)} \left [\log\frac{ p(x, y, z)}{p(y)q(z|x, y)} \right]  \\
\mathbb E_{z\sim q(.|x,y)} \left [\log\frac{ p(x, y, z)}{p(y)q(z|x,y)} \right]& =
\mathbb E_{z\sim q(.|x, y)} \left [\log{ p(x| y,z) } \right] -  D_{\mathrm{KL}}(q(z|x, y)\ \|\  p(z|y))
\end{aligned}
$$



## VAE

假设我们希望对分布$p(x)$采用，一个方案是先对$q(z)$采用，然后对$p(x|z)$采样，根据ELBO的结果，我们希望最大化：
$$
\log p(x)\ge \mathbb E_{z\sim q(.|x)} \left [\log\frac{ p(x, z)}{q(z|x)} \right]
=\mathbb E_{z\sim q(.|x)} \left [\log{ p(x| z) } \right] -  D_{\mathrm{KL}}(q(z|x)\ \|\  p(z))
$$
我们最大化下界：
$$
\mathbb E_{z\sim q(.|x)} \left [\log{ p(x| z) } \right] -  D_{\mathrm{KL}}(q(z|x)\ \|\  p(z))
$$
我们假设$q$是由$\phi$参数化，$p$是由$\theta$参数化，那么我们考虑如下优化问题：
$$
\max_{\phi, \theta} \mathcal L (\theta, \phi; x),  \\
\mathcal L (\theta, \phi; x) =\mathbb E_{z\sim q_{\phi}(.|x)} \left [\log{ p_{\theta}(x| z) } \right] -  D_{\mathrm{KL}}(q_{\phi}(z|x)\ \|\  p_{\theta}(z))
$$
接下来就是如何优化上式。

首先采样是无法求导的，所以要对$q_{\phi}(.|x)$分布做一些假设：假设$z\sim q_{\phi}(.|x)$等价于先采样$\epsilon \sim r(\epsilon)$，$z=f_{\phi}(\epsilon, x)$，此时（其中$n$为采样的个数）：
$$
\mathbb E_{z\sim q_{\phi}(.|x)} \left [\log{ p_{\theta}(x| z) } \right]
\approx \sum_{k=1}^n \log{ p_{\theta}(x| f_{\phi}(\epsilon_k, x)) },\epsilon_k \sim r(.),z_k=f_{\phi}(\epsilon_k, x).
$$
第二点是如何计算KL这一项，这里VAE假设$q_{\phi}(z|x),  p_{\theta}(z)$是高斯分布，并且：
$$
q_{\phi}(z|x)= \mathcal N(\mu_x , \sigma_x^2 I_d),\\
p_{\theta}(z)=\mathcal N(0, I_d)
$$
注意到此时可以利用如下结论（一维高斯分布）：
$$
\begin{aligned}
D_{\mathrm{KL}}(\mathcal N(\mu_1, \sigma_1^2) \ \|\  \mathcal N(\mu_2, \sigma_2^2))
&= \int \frac{1}{\sqrt{2\pi \sigma_1^2}} \exp\left(-\frac{(x-\mu_1)^2}{2 \sigma_1^2}  \right)
\left (
\log \frac{\sqrt{2\pi \sigma_2^2}}{\sqrt{2\pi \sigma_1^2}}
\exp\left (
-\frac{(x-\mu_1)^2}{2 \sigma_1^2} + \frac{(x-\mu_2)^2}{2 \sigma_2^2}
\right)
\right)\\
&= \int \frac{1}{\sqrt{2\pi \sigma_1^2}} \exp\left(-\frac{(x-\mu_1)^2}{2 \sigma_1^2}  \right)
\left ( \frac 1 2 \log \frac{\sigma_2^2}{\sigma_1^2}-\frac{(x-\mu_1)^2}{2 \sigma_1^2}
+ \frac{(x-\mu_2)^2}{2 \sigma_2^2}
\right) \\
&=\frac 1 2 \left(
-\log \sigma_1^2 + \log \sigma_2^2
- \mathbb E_{x\sim  \mathcal N(0, 1)}\left[
x^2 +\frac{(\sigma_1 x+\mu_1 -\mu_2)^2}{\sigma_2^2}
\right]
\right) \\
&=\frac 1 2 \left(
-\log \sigma_1^2 + \log \sigma_2^2
- \mathbb E_{x\sim  \mathcal N(0, 1)}\left[
\left (1+\frac{\sigma_1^2}{\sigma_2^2} \right)x^2 +
\frac{2\sigma_1(\mu_1 -\mu_2)}{\sigma _2^2} x + \frac{(\mu_1 -\mu_2)^2}{\sigma_2^2}
\right]
\right)\\
&= \frac 1 2 \left(
-\log \sigma_1^2 + \log \sigma_2^2
- 1+\frac{\sigma_1^2}{\sigma_2^2} +\frac{(\mu_1 -\mu_2)^2}{\sigma_2^2}
\right)
\end{aligned}
$$
将$\mu_2=0, \sigma_2=1$代入可得：
$$
\begin{aligned}
D_{\mathrm{KL}}(\mathcal N(\mu_x, \sigma_x^2) \ \|\  \mathcal N(0, 1))
&= \frac 1 2 \left(
-\log \sigma_x^2
+{\sigma_x^2} +\mu_x ^2- 1
\right)
\end{aligned}
$$
推广到高维可得：
$$
\begin{aligned}
D_{\mathrm{KL}}(\mathcal N(\mu, \sigma^2) \ \|\  \mathcal N(0, I_d))
&= \sum_{i=1}^d\frac 1 2 \left(
-\log \sigma_i^2
+{\sigma_i^2} +\mu_i ^2- 1
\right)
\end{aligned}
$$
现在带回公式18可得：
$$
\begin{aligned}
\mathcal L (\theta, \phi; x)
& =\mathbb E_{z\sim q_{\phi}(.|x)} \left [\log{ p_{\theta}(x| z) } \right] -  D_{\mathrm{KL}}(q_{\phi}(z|x)\ \|\  p_{\theta}(z)) \\
& \approx
\sum_{k=1}^n \log{ p_{\theta}(x| f_{\phi}(\epsilon_k, x)) }
+\frac 1 2 \sum_{i=1}^d \left(
1 + \log \sigma_i^2
-\mu_i ^2
-{\sigma_i^2}
\right)
\end{aligned}
$$
将假设：
$$
q_{\phi}(z|x)= \mathcal N(\mu_x , \sigma_x^2 I_d),z=\sigma_x \epsilon + \mu_x , \epsilon \sim r(.)
$$
带入可得：
$$
\begin{aligned}
\mathcal L (\theta, \phi; x)
 & \approx
\sum_{k=1}^n \log{ p_{\theta}(x| f_{\phi}(\epsilon_k, x)) }
+\frac 1 2 \sum_{i=1}^d \left(
1 + \log \sigma_i^2
-\mu_i ^2
-{\sigma_i^2}
\right) \\
& = \sum_{k=1}^n \log{ p_{\theta}(x| z_k) }
+\frac 1 2 \sum_{i=1}^d \left(
1 + \log \sigma_i^2
-\mu_i ^2
-{\sigma_i^2}
\right) \\
z_k &=\sigma_{x} \epsilon_k + \mu_x , \epsilon \sim r(.),
\mu_x = [\mu_1,\ldots, \mu_d]^\top, \sigma_x =
[\sigma_1,\ldots, \sigma_d]
\end{aligned}
$$


### 关于Loss

#### 连续情形

假设$x$为连续变量，我们假设
$$
x| z  \sim \mathcal N(\mu_z, \Sigma_z)

$$
因此：
$$
p_{\theta}(x| z ) =\frac{1}{\sqrt{(2\pi)^d |\Sigma_z|}}
\exp\left( -\frac 1 2 (x-\mu_z)^\top \Sigma_z^{-1}(x-\mu_z)  \right)\\
\log p_{\theta}(x| z ) = -\frac 12
\left(
\log |\Sigma_z| + d\log(2\pi) + (x-\mu_z)^\top \Sigma_z^{-1} (x-\mu_z)
\right)
$$
进一步简化，假设$\Sigma_z= I_d$，那么：
$$
\log p_{\theta}(x| z ) = -\frac 12
\left(
  d\log(2\pi) + (x-\mu_z)^\top  (x-\mu_z)
\right)
$$
因为第一项为常数，所以损失函数变成：
$$
\begin{aligned}
\mathcal L (\theta, \phi; x)

& = -\frac 1 2\sum_{k=1}^n  (x-\mu_z)^\top  (x-\mu_z)
+\frac 1 2 \sum_{i=1}^d \left(
1 + \log \sigma_i^2
-\mu_i ^2
-{\sigma_i^2}
\right) \\
z_k &=\sigma_{x} \epsilon_k + \mu_x , \epsilon \sim r(.),
\mu_x = [\mu_1,\ldots, \mu_d]^\top, \sigma_x =
[\sigma_1,\ldots, \sigma_d]
\end{aligned}
$$
训练流程：

- 输入$x$；
- 根据encoder计算$\sigma_x, \mu_x$；
- 采样$z_k =\sigma_{x} \epsilon_k + \mu_x $；
- 根据decoder计算$\mu_z$；
- 根据公式74计算loss；

推理：

- 随机采样$z$；
- 利用decoder计算$\mu_z$得到输出；

参考资料：

- https://stats.stackexchange.com/questions/323568/help-understanding-reconstruction-loss-in-variational-autoencoder



#### 离散情形

假设$x$为离散变量，我们假设
$$
x| z  \sim \text{Multinomial}( p), p=[p_1, p_2, \ldots, p_k]
$$
那么此时：
$$
\log p_{\theta}(x| z ) =
\sum_{i=1}^d  \sum_{j=1}^k \mathbf 1_{x_i=j} \log p_j
$$
损失函数变成：
$$
\begin{aligned}
\mathcal L (\theta, \phi; x)

& = \sum_{i=1}^d  \sum_{j=1}^k \mathbf 1_{x_i=j} \log p_j
+\frac 1 2 \sum_{i=1}^d \left(
1 + \log \sigma_i^2
-\mu_i ^2
-{\sigma_i^2}
\right) \\
z_k &=\sigma_{x} \epsilon_k + \mu_x , \epsilon \sim r(.),
\mu_x = [\mu_1,\ldots, \mu_d]^\top, \sigma_x =
[\sigma_1,\ldots, \sigma_d]
\end{aligned}
$$
训练流程：

- 输入$x$；
- 根据encoder计算$\sigma_x, \mu_x$；
- 采样$z_k =\sigma_{x} \epsilon_k + \mu_x $；
- 根据decoder计算$p=\mathrm{Softmax}(f(z_k))$；
- 根据公式77计算loss；

推理：

- 随机采样$z$；
- 利用decoder计算$p=\mathrm{Softmax}(f(z_k))$，然后采样得到输出；



### VAE的限制

注意VAE采样的时候是并行生成的，这其实是假设每个像素独立。



## CVAE

根据ELBO推广，可以很容易得到condition VAE的loss：
$$
\begin{aligned}
\mathcal L (\theta, \phi; x)

& = -\frac 1 2\sum_{k=1}^n  (x-\mu_{z,y})^\top  (x-\mu_{z,y})
+\frac 1 2 \sum_{i=1}^d \left(
1 + \log \sigma_i^2
-\mu_i ^2
-{\sigma_i^2}
\right) \\
z_k &=\sigma_{x,y} \epsilon_k + \mu_{x,y} , \epsilon \sim r(.),
\mu_{x,y} = [\mu_1,\ldots, \mu_d]^\top, \sigma_{x,y} =
[\sigma_1,\ldots, \sigma_d]
\end{aligned}
$$
