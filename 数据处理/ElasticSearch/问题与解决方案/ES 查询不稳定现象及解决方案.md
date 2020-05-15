【问题描述】：对于相同的查询语句，ES 查询返回不同的检索内容。

【示例 1】：{"filter": {}, "match": {"content": "宝骏510"}, "order": "score", "size": 10}。

![示例1](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL2NsdnNpdC9tYXJrZG93bi1pbWFnZS9tYXN0ZXIvZGF0YS9lcy8yMDIwMDQxNjIzMDgyMi5wbmc?x-oss-process=image/format,png)![示例2](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL2NsdnNpdC9tYXJrZG93bi1pbWFnZS9tYXN0ZXIvZGF0YS9lcy8yMDIwMDQxNjIzMDc0MC5wbmc?x-oss-process=image/format,png)

在上面的示例中可以看到推荐句子的第二句和第三句的顺序发生了变化。

## 产生原因
【官方文档】：Scores are not reproducible
- Say the same user runs the same request twice in a row and documents do not come back in the same order both times, this is a pretty bad experience isn’t it? Unfortunately this is something that can happen if you have replicas (index.number_of_replicas is greater than 0). The reason is that Elasticsearch selects the shards that the query should go to in a round-robin fashion, so it is quite likely if you run the same query twice in a row that it will go to different copies of the same shard.
- Now why is it a problem? Index statistics are an important part of the score. And these index statistics may be different across copies of the same shard due to deleted documents. As you may know when documents are deleted or updated, the old document is not immediately removed from the index, it is just marked as deleted and it will only be removed from disk on the next time that the segment this old document belongs to is merged. However for practical reasons, those deleted documents are taken into account for index statistics. So imagine that the primary shard just finished a large merge that removed lots of deleted documents, then it might have index statistics that are sufficiently different from the replica (which still have plenty of deleted documents) so that scores are different too.

【中文翻译】：
- 假设同一位用户连续两次执行相同的请求，但文档两次都没有以相同的顺序返回，这是非常糟糕的体验。不幸的是，如果您的 ES 存在副本，则可能会发生这种情况。ElasticSearch 以轮询的方式选择查询操作时访问的分片，因此如果连续运行两次相同的查询，则很有可能会访问同一分片的不同副本。
- 现在为什么会出现问题？索引统计是分数的重要组成部分，并且由于删除的文档，同一分片的副本之间的索引统计可能会有所不同。ES 在删除或更新文档时不会立即将旧文档从索引中删除，而是将其标记为已删除，并且仅在下次合并旧文档所属的段时才从磁盘中删除它们（旧文档）。但是，由于实际原因（参考下面的假设），这些已删除的文档仍然会参与到索引统计中。假设主分片刚刚完成了一个大型合并，删除了许多已删除的文档，那么它的索引统计信息可能与副本（仍具有大量已删除文档）完全不同，因此得分也有所不同。

## 解决方案
【官方文档】：
- The recommended way to work around this issue is to use a string that identifies the user that is logged is (a user id or session id for instance) as a preference. This ensures that all queries of a given user are always going to hit the same shards, so scores remain more consistent across queries.
- This work around has another benefit: when two documents have the same score, they will be sorted by their internal Lucene doc id (which is unrelated to the _id) by default. However these doc ids could be different across copies of the same shard. So by always hitting the same shard, we would get more consistent ordering of documents that have the same scores.

【中文翻译】：
- 解决此问题的推荐方法：使用一个字符串来标识已登录用户（例如用户ID或会话ID）作为首选项。这样可以确保给定用户的所有查询始终会 hit 到相同的分片，从而使各个查询的分数保持一致。
- 解决此问题的另一个好处：当两个文档的分数相同时，默认情况下将按其内部的 Lucene doc id（与 _id 无关）进行排序。但是，相同分片的副本之间的文档 ID 可能有所不同。因此，通过始终 hit 相同的分片，相同分数的文档的排序会更稳定。

【实际方案】：最终并没有采用官方给出的方案，在查询 API 新增了一个参数，用以指定每次查询的切片。
```python
hits = self.es.search(self.index, body=query, preference="_primary")['hits']['hits']
```
如上述代码所示，在 search() 函数中新增 preference 参数，指明每次查询的切片为主切片。方案参考：

【Python elasticsearch 文档地址】：https://elasticsearch-py.readthedocs.io/en/master/api.html#elasticsearch

## 参考资料
- es 相同条件 执行两次 不同结果 ,求解：https://elasticsearch.cn/question/5186
- 官方文档：https://www.elastic.co/guide/en/elasticsearch/reference/master/consistent-scoring.html