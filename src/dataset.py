
# 23个声母，24个韵母
initials = 'b p m f d t n l ɡ k h j q x zh ch sh r z c s y w'.split(' ')
initial2id = {initial:i+1 for i, initial in enumerate(initials)}

finals = 'a o e i u v ai ei ui ao ou iu ie ue er an en in un vn ang eng ing ong'.split(' ')
final2id = {final:i for i, final in enumerate(finals)}

from pypinyin import lazy_pinyin, Style
print(lazy_pinyin('绿色，',style = Style.TONE3))