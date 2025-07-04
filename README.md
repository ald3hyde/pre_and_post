# dump2glassanalysis.py
 - lammpsのMDシミュレーションで得られたdump (.lammpstrj) ファイルから
   配位数とX-O-X (X: ネットワークフォーマー) の計算を行うプログラム。
# dump2glassdescript.py
 - lammpsのMDシミュレーションで得られたdumpファイルから，配位数，$Q^n$
   ，ガラス構造を表現する構造記述子を計算するプログラムです。
 - スラブがある系で表面のみに着目したい場合は，最も露出しているSiから
   の距離 (CUTOFF (default: 5.0 Å))をプログラムで指定して，-s オプショ
   ンを追加する。
# 詳細な情報
- https://amorphous.tf.chiba-u.jp/kayano/ まで。
