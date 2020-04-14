from subjects import subjects

marker_default = '✓'
n_done = 40

with open("README.md", mode='w') as f:
    f.write("# Image-Processing\n\n")
    f.write("[画像処理100本ノック](https://qiita.com/yoyoyo_/items/2ef53f47f87dcf5d1e14)をやってみる。\n\n")

    f.write("## 進捗\n\n")
    f.write("| 番号 | 内容 | pyhtonの解答 | 出力画像 |\n")
    f.write("|:----:|:----:|:----:|:----:|\n")

    for i, subject in enumerate(subjects):
        marker = marker_default if i < n_done else ""
        link_code = "[{}](https://github.com/HirokiNishimoto/Image_Processing/blob/master/solve_python/solve{}.py)".format(marker, str(i+1).zfill(2))
        link_png = "[{}](https://github.com/HirokiNishimoto/Image_Processing/blob/master/img/out/q_{}.png)".format(marker, str(i+1).zfill(2))
        f.write("| {} | {} | {} | {} |\n".format(i+1, subject, link_code, link_png))

    f.write("\n## 参考および引用元\n[画像処理100本ノックを作ったった(Qiita)](https://qiita.com/yoyoyo_/items/2ef53f47f87dcf5d1e14) <br>[画像処理100本ノック!!(github)](https://github.com/yoyoyo-yo/Gasyori100knock) <br>\n")
