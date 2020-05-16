
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


## 索引
文档可以通过 `index` API 被索引——**存储**并使其可**搜索**。

### 文档存储
文档通过 `_index`、`_type` 以及 `_id` 来确定存储的唯一性。同一类事物的 `_index` 和 `_type` 都是相同的，因此通过 `_id` 来唯一标识文档。

`_id` 有两种方式：
- 使用自己的 ID；
- 自增 ID。

#### 使用自己的 ID
如果文档拥有天然的标识符，例如用户类中的用户 ID，这时就可以直接用用户 ID 来作为 `_id`。

【示例】：
- 请求内容

```
PUT /website/user/123
{
    "userId": 123,
    "userName": "XXX"
}
```
- 返回内容

```
{
    "`_index": "website",
    "`_type":  "user",
    "`_id":    "123",
    "_version": 1,
    "created":  true
}
```
上述返回值意味着索引请求已经被成功创建，其中还包含了 `_index`、`_type` 以及 `_id` 的元数据，以及一个新的元素 `_version`。

在 ES 中，每个文档都有一个版本号码，每当文档产生变化时（包括删除），`_version` 就会增大。

#### 自增 ID
如果数据中没有天然的标识符，我们可以让 ES 为我们自动生成一个。

【示例】：
- 请求内容：自动生成 ID，因此不需要填写 `_id`，所以请求中只包含 `_index` 和 `_type`。

```
POST /website/user/
{
    "userId": 123,
    "userName": "XXX"
}
```

- 返回内容：与之前基本一样，只有 `_id` 改成了系统生成的自增值。

```
{
    "`_index": "website",
    "`_type":  "user",
    "`_id":    "wM0OSFhDQXGZAWDf0-drSA",
    "_version": 1,
    "created":  true
}
```

请求的结构发生了变化，把 `PUT` 替换为了 `POST`。
- `PUT`：把文档存储在这个地址中。
- `POST`：把文档存储在这个地址下。

【注意】：自生成 ID 是由 22 个字母组成的，被称为 [UUIDs](http://baike.baidu.com/view/1052579.htm?fr=aladdin)。

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

