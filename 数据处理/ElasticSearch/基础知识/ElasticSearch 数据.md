
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
    "_index":   "website",
    "_type":    "user",
    "_id":      "wM0OSFhDQXGZAWDf0-drSA",
    "_version": 1,
    "created":  true
}
```

请求的结构发生了变化，把 `PUT` 替换为了 `POST`。
- `PUT`：把文档存储在这个地址中。
- `POST`：把文档存储在这个地址下。

【注意】：自生成 ID 是由 22 个字母组成的，被称为 [UUIDs](http://baike.baidu.com/view/1052579.htm?fr=aladdin)。

## 搜索文档
要从 ES 中获取文档，需要先知道文档在 ES 中的位置。根据先前所学的知识，`_index`、`_type` 和 `_id` 唯一标识文档，因此，我们可通过这三者去获取文档。

【示例】：
- 请求内容

```
GEt /website/user/123?pretty
```
- 返回内容

```
{
    "_index":   "website",
    "_type":    "user",
    "_id":      "wM0OSFhDQXGZAWDf0-drSA",
    "_version": 1,
    "found":    true
    "_source": {
        "userId": 123,
        "userName": "XXX"
    }
}
```

`_source` 表示存储文档时传入的内容，即上一节**文档存储**中插入的内容。

【pretty】：在任意查询字符串中添加 `pretty` 参数，ES 可以执行**优美打印**，得到更加易于识别的 JSON 结果。`_source` 字段不会执行优美打印，只保留录入文档时的样子。

【found】：GET 请求的返回结果中包含 `found` 字段，表示文档是否有被检索得到。如果我们请求了一个不存在的文档，`found` 值变为 false，并且 HTTP 返回码也会变为 "404 Not Found"。

通常，`GET` 请求会将整个文档一并放入 `_source` 字段中。如果此时我们只需要特定字段，可以使用 `_source` 指定需要返回的字段。

【示例】：
```
GET /website/user/123?_source=userId,userName
```

多个字段可以使用逗号分隔。

如果只想得到 `_source` 字段而不想返回 `_index` 等其他元数据，可以这样请求：
```
GET /website/user/123/_source
```

这样结果只返回 `_source` 字段中的内容。
```
{
    "userId": 123,
    "userName": "XXX"
}
```

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

## 创建文档
当我们在索引一个文档时，如何确定是创建了一个新的文档，还是覆盖了一个已经存在的文档？

如果不考虑覆盖的情况，最简单的方法就是使用 `POST`，让 ES 创建不同的 `_id`，不考虑覆盖直接创建文档，即使文档内容相同。如果使用文档中已有 id 信息，那么会覆盖原先的文档内容。

但很多时候我们往往会有去重的任务，此时 ID 就可以帮助我们做去重任务。只有 `_index`、`_type` 和 `_id` 这三个元数据不完全相同才允许插入。实现这个目的有两种方法：
- 在查询中添加 `op_type` 参数：

```
PUT /website/user/123?op_type=create
```
- 在请求最后添加 `/create`：

```
PUT /website/user/123/_create
```

如果成功创建了新的文档，ES 将会返回常见的元数据以及 `201 Created` HTTP 返回码。如果存在同名文档，ES 将会返回 `409 Conflict` 的 HTTP 返回码。