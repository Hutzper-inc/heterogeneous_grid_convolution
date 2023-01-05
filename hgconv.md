## HGConv
全体の式は以下の通り
$$ Z = \tilde{S} \sum_{\delta \in \Delta}(\hat{D}^\delta)^{-1} \hat{A}^\delta \bar{S}^TXW_{\delta}$$ 
### init
まずカーネルサイズからグリッドの隣接行列を作成しておく
networkxを使ってチャチャっと作成
$$ A^{\delta} : 隣接行列$$

### Clustering
API使ってグルーピングする。
この時、xは(B, C, H, W)
$$S = ssn iter(x)$$
$$ S \in R(B, N_{pix}, N_{grp})$$

### Pooling
列(グループ)方向の合計が1になるようにNormalizeする
$$ \bar{S} = S\bar{Z}^{-1}$$
$$ \hat{X} = \bar{S}^TX$$

### Graph Conv
まず、Noise Reductionで隣接行列をきれいにする
$$ \hat{A}^\delta = S^T A^\delta S $$
$$ \hat{A}^\delta = Relu(\hat{A}^\delta - \hat{A}^\delta_{opposit} )$$
ここで
$$ \hat{A}^\delta \in R(N_{grp}, N_{grp})$$
そんでスケーリングマトリックスも同時に計算する。
スケーリングマトリックスは対角行列になる。
Aは有向行列なのでどっちにsumとればいいんだ...?多分列方向。
$$ \hat{D}^\delta = degree(\hat{A}^\delta)$$