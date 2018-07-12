from mrjob.job import MRJob


class MRWordCounter(MRJob):
    def mapper(self, key, line):
        """
        返回每一个单词和其对应的值
        :param key:
        :param line:
        :return:
        """
        for word in line.split():
            yield word, 2

    def reducer(self, word, occurrences):
        """
        将相同的单词的值相加起来，返回的结果就是所有的单词出现的次数
        :param word:
        :param occurrences:
        :return:
        """
        yield word, sum(occurrences)


if __name__ == "__main__":
    MRWordCounter.run()