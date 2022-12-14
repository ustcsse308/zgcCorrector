import re


def stastics(sents1, sents2, sents3):
    """
    该纠的，即有错文本记为 P，不该纠的，即无错文本记为 N
    对于该纠的，纠对了，记为 TP，纠错了或未纠，记为 FP
    对于不该纠的，未纠，记为 TN，纠了，记为 FN。
    :param sents1: input
    :param sents2: output
    :param sents3: target
    :return: F_sent
    """

    TP, TN, FP, FN = 0, 0, 0, 0
    TP_sent, TN_sent, FP_sent, FN_sent = 0, 0, 0, 0

    for sent1, sent2, sent3 in zip(sents1, sents2, sents3):
        sent1 = re.sub("[zxcvbnmlkjhgfdsaqwertyuiopZXCVBNMASDFGHJKLQWERTYUIOP"
                       "／,，‘”〝〞（“ ）＊×〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】。，、＇：∶；?ˆˇ﹕︰"
                       "﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎"
                       "+=<＿_-\ˇ~﹉﹊aa︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼]", "", sent1)
        sent1 = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+", "", sent1)
        sent2 = re.sub("[zxcvbnmlkjhgfdsaqwertyuiopZXCVBNMASDFGHJKLQWERTYUIOP"
                       "／,，‘”〝〞（“ ）＊×〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】。，、＇：∶；?ˆˇ﹕︰"
                       "﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎"
                       "+=<＿_-\ˇ~﹉﹊aa︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼]", "", sent2)
        sent2 = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+", "", sent2)
        sent3 = re.sub("[zxcvbnmlkjhgfdsaqwertyuiopZXCVBNMASDFGHJKLQWERTYUIOP"
                       "／,，‘”〝〞（“ ）＊×〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】。，、＇：∶；?ˆˇ﹕︰"
                       "﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎"
                       "+=<＿_-\ˇ~﹉﹊aa︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼]", "", sent3)
        sent3 = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+", "", sent3)
        # if sent1 == sent2 and sent1 == sent3:
        #     continue
        if sent1 != sent3:
            if sent3 == sent2:
                TP_sent += 1
            else:
                FP_sent += 1
        else:
            if sent1 == sent2:
                TN_sent += 1
            else:
                FN_sent += 1
        if len(sent1) == len(sent2) and len(sent1) == len(sent3):
            for i in range(len(sent1)):
                if sent1[i] != sent3[i]:
                    if sent3[i] == sent2[i]:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if sent1[i] == sent2[i]:
                        TN += 1
                    else:
                        FN += 1

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    beta = 1
    F = (1 + beta ** 2) * precision * recall / ((beta ** 2 * precision) + recall)
    accuracy = round(accuracy, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    F = round(F, 4)
    print('char level...')
    print("valid chars:\t{}\nacc:\t{}\npre:\t{}\nrecall:\t{}\nF:\t{}\n".format(TP + FP + TN + FN, accuracy, precision,
                                                                     recall, F))
    print("TP:\t{}\nFP:\t{}\nfn:\t{}\nTN:\t{}".format(TP, FP, FN, TN))

    accuracy_sent = (TP_sent + TN_sent) / (TP_sent + FP_sent + FN_sent + TN_sent)
    precision_sent = TP_sent / (TP_sent + FP_sent)
    recall_sent = TP_sent / (TP_sent + FN_sent)
    F_sent = (1 + beta ** 2) * precision_sent * recall_sent / ((beta ** 2 * precision_sent) + recall_sent)
    accuracy_sent = round(accuracy_sent, 4)
    precision_sent = round(precision_sent, 4)
    recall_sent = round(recall_sent, 4)
    F_sent = round(F_sent, 4)

    print()
    print('sentence level...')
    print("valid sentences:\t{}\nacc:\t{}\npre:\t{}\nrecall:\t{}\nF:\t{}\n".format(TP_sent + FP_sent + TN_sent + FN_sent,
                                                                         accuracy_sent, precision_sent, recall_sent,
                                                                         F_sent))
    print("TP:\t{}\nFP:\t{}\nfn:\t{}\nTN:\t{}".format(TP_sent, FP_sent, FN_sent, TN_sent))
    return F_sent

