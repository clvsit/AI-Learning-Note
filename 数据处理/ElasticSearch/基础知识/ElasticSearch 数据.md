
## 文档元数据
一个文档不只包含了数据，还包含了关于文档的信息，它们被称为元数据（metadata）。

元数据中有三个元素是必须存在的：
- `_type`：文档代表的对象种类。
- `_index`：文档存储的地方。
- `_id`：文档的唯一编号。

### `_index`
索引类似于传统数据库中的“数据库”——也就是我们存储并索引相关数据的地方。

在 ES 中，我们的数据都在存储**分片**中，因此 `index`（索引）只是一个逻辑命名空间，它可以将一个或多个分片组合在一起。

索引名称必须全部小写，不能以下划线开头，且不能包含逗号。

### `_type`
文档代表的对象类型，定义了对象的属性或者与数据的关联。每一个类型都拥有自己的映射（mapping），定义了当前对象类型下的数据结构，类似于传统数据库表的列，表明每个字段的数据结构。

### `_id`
`_id` 唯一标识文档，与 `_index`、`_type` 组合使用时，可以代表 ES 中一个特定的文档。




## 检查文档是否存在
可以使用 `HEAD` 方法，通过判断返回的 HTTP 头文件来检查文档是否存在。
```
curl -i XHEAD /website/blog/123
```
- 文档存在：

```
HTTP/1.1 200 OK
Content-Type: text/plain; charset=UTF-8
Content-Length: 0
```
- 文档不存在：

```
HTTP/1.1 404 Not Found
Content-Type: text/plain; charset=UTF-8
Content-Length: 0
```

## 更新文档
文档是不可改变的，如果需要改变，则使用 `index` API 来重新索引或者替换。
```
PUT /website/blog/123
{
    "title": "My first blog entry",
    "text":  "I am starting to get the hang of this...",
    "date":  "2014/01/02"
}
```

在反馈中，可以发现 ES 已经将 `_version` 数值增加了。
```
{
    "_index":   "website",
    "_type":    "blog",
    "_id":      "123",
    "_version":  2,
    "created":   false <1>
}
```
`created` 字段被标记为 `false` 是因为在同索引、同类型下已经存在同 ID 的文档。

在内部，ES 已经将旧文档标记为删除，并添加了新的文档。需要注意的是，旧的文档不会立即消失，但无法访问，ES 后续会在后台清理已经删除的文件。

