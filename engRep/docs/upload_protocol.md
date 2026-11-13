1. 文档目的

本文档定义数据生产方将批次数据发布到 OSS 时应遵循的协议，包括 OSS 目录、gzip 数据文件、批次 Manifest、业务级滚动 Meta 以及上传顺序。

消费方会根据本协议发现完整批次、校验文件，并按照批次和分片顺序处理数据。因此，生产方必须保证：

一个业务对应一份独立的滚动 Meta；
数据文件使用 gzip 压缩；
批次版本严格递增且唯一；
数据文件全部上传成功后才能发布 Manifest；
Manifest 发布成功后才能把批次加入滚动 Meta；
已发布的数据文件和 Manifest 不得覆盖或修改。

本文只约定生产方需要实现的发布协议，不约定消费方内部的 Elasticsearch、DolphinScheduler 或 MongoDB 实现。

2. 通用约定
2.1 业务标识

businessCode 用于唯一标识一个数据业务，也对应消费方的一套独立导入流程。例如：



book-online-sources
book-works


不同 businessCode 必须使用不同的 OSS 目录和 meta.json，互不混用。

2.2 批次版本

batchVersion 是一个业务内批次的唯一版本号，并且必须严格单调递增。建议格式为：



yyyyMMddHHmmssSSS-序号


示例：



20260915100000123-0001


约束如下：

使用 UTC 时间生成时间部分；
同一个 businessCode 下不得重复；
新批次必须大于所有已发布批次；
必须可以直接按照字符串升序排序；
生成批次版本时不能只依赖秒级时间，以免同一秒内发生冲突；
已发布的 batchVersion 不得重新用于另一批数据。
2.3 时间与字符编码
JSON 和 JSONL 文件统一使用 UTF-8 编码，不带 BOM；
JSON 时间字段使用 ISO 8601 UTC 格式，例如 2026-09-15T10:00:00Z；
文件大小单位为字节；
文件大小上限中的 MiB 表示 1024 × 1024 字节。
一、OSS 目录与文件结构
1.1 推荐目录结构




/{environment}/{businessCode}/
├── meta.json
└── batches/
    ├── {batchVersion}/
    │   ├── manifest.json
    │   ├── data_part0000.jsonl.gz
    │   ├── data_part0001.jsonl.gz
    │   └── ...
    └── {nextBatchVersion}/
        ├── manifest.json
        └── data_part0000.jsonl.gz


示例：



/prod/book-online-sources/meta.json
/prod/book-online-sources/batches/20260915100000123-0001/manifest.json
/prod/book-online-sources/batches/20260915100000123-0001/data_part0000.jsonl.gz
/prod/book-online-sources/batches/20260915100000123-0001/data_part0001.jsonl.gz


目录和文件职责：

对象	是否可覆盖	用途
meta.json	是	保存该业务保留期内已经完整发布的批次列表
manifest.json	否	描述单个批次的完整文件清单，是该批次的完成标记
data_partNNNN.jsonl.gz	否	保存批次业务变更记录
1.2 文件命名规则

数据分片统一命名为：



data_part0000.jsonl.gz
data_part0001.jsonl.gz
data_part0002.jsonl.gz


要求：

part 序号从 0000 开始；
序号连续，不允许跳号或重复；
文件名中的顺序必须与 Manifest 中的 sequence 一致；
文件名区分大小写，生产方和消费方不得自行转换大小写；
OSS 对象路径必须使用 /，不得使用系统本地路径分隔符。
1.3 环境隔离

开发、测试、预发和生产环境必须使用不同的 OSS 前缀，例如：

/dev/book-online-sources/
/test/book-online-sources/
/prod/book-online-sources/


生产方不得把不同环境的数据写入同一业务目录。

二、数据文件格式
2.1 文件格式

数据文件使用 gzip 压缩的 JSON Lines 格式：



data_partNNNN.jsonl.gz


解压后，每一行是一条完整且独立的 JSON 变更记录。示例：



{"op":"upsert","id":"uuid-1","data":{"bookSourceId":"uuid-1","siteCode":"qidian","siteBookId":"book_12345","bookUnifiedId":"uuid-w1","title":"星辰变","primaryAuthor":"我吃西红柿","coverUrl":"https://example.com/cover.jpg","description":"作品简介","serialStatus":"serializing","totalWordCount":5000000}}
{"op":"upsert","id":"uuid-2","data":{"bookSourceId":"uuid-2","coverUrl":"https://example.com/new-cover.jpg","description":"新版简介"}}
{"op":"delete","id":"uuid-3"}


通用记录结构：

字段	类型	必填	说明
op	string	是	操作类型，第一版仅允许 upsert 或 delete
id	string	是	记录唯一标识，后续重复发布同一条业务数据时必须保持稳定
data	object	upsert 时必填	需要新增或更新的业务字段；delete 时不得出现
2.2 操作语义
upsert

upsert 表示：目标记录不存在时创建，存在时更新 data 中提供的字段。

生产方必须遵守以下字段语义：

字段未出现在 data 中：表示本次不修改该字段；
字段明确设置为 null：表示本次把该字段置空；
data 必须是 JSON 对象，不得为 null、数组或字符串；
data 中不得包含消费存储系统的控制字段；
id 与业务数据中的主键字段如果同时存在，两者必须一致。

例如：



{"op":"upsert","id":"uuid-1","data":{"coverUrl":null}}


表示明确清空 coverUrl，不是保持原值。

delete

delete 表示删除 id 对应的目标记录。删除一条不存在的记录也应视为幂等操作，生产方不需要为此生成补偿数据。

示例：



{"op":"delete","id":"uuid-3"}

2.3 为什么数据文件不直接使用 Elasticsearch _bulk 格式

生产协议使用业务变更结构，而不是直接输出 Elasticsearch _bulk 请求，主要有以下好处：

生产方不依赖 Elasticsearch 实现细节。 生产方只表达“新增或更新”“删除”等业务含义，不需要了解 index、update、doc_as_upsert 等 Elasticsearch 参数。
存储系统可以独立演进。 消费方将来调整索引名、路由、版本控制、写入参数，甚至更换存储系统时，不需要数据生产方重新生成协议文件。
避免生产方生成错误的 Bulk 动作。 _bulk 对动作行和数据行的组合有严格要求，直接生成容易出现动作名或行数不匹配等问题。例如 {"upsert":{"_id":"..."}} 并不是合法的 Elasticsearch Bulk 动作。
业务语义更清晰。 一条 JSONL 就是一条变更记录，recordCount 可以直接按行统计，不需要根据不同 Bulk 动作判断一条还是两条物理行。
便于校验、回放和排障。 文件可以在不连接 Elasticsearch 的情况下完成格式校验，也可以被其他消费者复用。
避免暴露消费端配置。 索引名、别名、路由规则和写入策略属于消费方配置，不需要写入跨网络交换的数据文件。

消费方负责把业务变更记录转换成目标存储系统所需的写入请求。该转换不属于本生产协议。

2.4 分片与压缩规则

生产方必须遵守以下规则：

使用标准 gzip 格式压缩，每个分片是一个可以独立解压的 gzip 文件；
每个 OSS gzip 对象压缩后不得超过 10 MiB；
建议以 8 MiB 左右作为压缩后目标大小，为 gzip 压缩率波动预留空间；
只能在完整记录之间切分，不得把一条 JSON 拆到两个文件；
解压后的每一行只能包含一个 JSON 对象；
JSON 对象内部不得包含未转义的物理换行；
每行末尾使用 \n，最后一行也应以 \n 结束；
不得生成空分片；
单条记录压缩后仍超过 10 MiB 时，整个批次发布失败，生产方应告警，不得拆分该记录；
压缩结束后必须检查最终对象大小，不能只根据未压缩大小估算是否符合上限。

为了保证结果稳定、便于对账，同一个批次中的数据应按照双方约定的业务键排序。图书数据建议排序键为：



bookUnifiedId ASC, bookSourceId ASC, id ASC


如果某类记录没有前面的业务字段，则继续使用后续可用字段排序。相同 id 在同一批次出现多次时，其先后顺序必须与业务变更顺序一致；更推荐生产方在发布前合并为该批次的最终变更。

2.5 内容约束
空批次不生成数据文件，也不发布 Manifest 和 Meta 条目；
同一个 id 跨批次出现是允许的；
批次之间的数据生效顺序由 batchVersion 决定；
recordCount 按业务变更记录数统计，即解压后的 JSONL 行数；
gzip 文件的扩展名必须为 .jsonl.gz；
生产方必须在上传前完成 JSON 语法、必填字段、操作类型和主键校验。
三、批次 Manifest
3.1 作用

每个批次必须包含一个不可变的 manifest.json。它同时承担以下职责：

声明批次基本信息；
给出完整、有序的数据文件清单；
提供文件大小、记录数和 SHA-256 校验值；
作为该批次所有数据文件已经上传完成的发布标记。

消费方只有在滚动 Meta 中看到批次，并成功读取对应 Manifest 后，才会处理该批次。

3.2 格式示例




{
  "protocolVersion": "1.0",
  "businessCode": "book-online-sources",
  "batchVersion": "20260915100000123-0001",
  "batchType": "incremental",
  "producedAt": "2026-09-15T10:00:00Z",
  "format": "jsonl",
  "compression": "gzip",
  "contentEncoding": "utf-8",
  "files": [
    {
      "sequence": 0,
      "fileName": "data_part0000.jsonl.gz",
      "sizeBytes": 8380123,
      "uncompressedSizeBytes": 18743291,
      "recordCount": 30000,
      "checksum": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    },
    {
      "sequence": 1,
      "fileName": "data_part0001.jsonl.gz",
      "sizeBytes": 4157821,
      "uncompressedSizeBytes": 9342280,
      "recordCount": 15000,
      "checksum": "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    }
  ],
  "totalFileCount": 2,
  "totalRecordCount": 45000
}

3.3 字段说明
字段	类型	必填	说明
protocolVersion	string	是	本协议版本，当前固定为 1.0
businessCode	string	是	业务唯一标识，必须与 OSS 业务目录一致
batchVersion	string	是	批次版本，必须与批次目录名称一致
batchType	string	是	第一版固定为 incremental
producedAt	string	是	批次生产完成时间，ISO 8601 UTC 格式
format	string	是	固定为 jsonl
compression	string	是	固定为 gzip
contentEncoding	string	是	固定为 utf-8
files	array	是	当前批次全部数据分片，按 sequence 升序排列
totalFileCount	integer	是	files 数组长度
totalRecordCount	integer	是	所有文件 recordCount 之和

单个 files 条目：

字段	类型	必填	说明
sequence	integer	是	分片序号，从 0 开始连续递增
fileName	string	是	当前批次目录下的数据文件名，不得包含 /、.. 或完整 URL
sizeBytes	integer	是	OSS 中 gzip 对象的实际字节数
uncompressedSizeBytes	integer	是	解压后 JSONL 内容的实际字节数
recordCount	integer	是	解压后的 JSONL 行数
checksum	string	是	gzip 对象完整字节内容的 SHA-256，格式为 sha256:<64位小写十六进制>
3.4 校验规则

生产方在发布 Manifest 前必须确认：

Manifest 中列出的全部对象已上传成功；
sequence 从 0 开始且连续；
fileName 与实际 OSS 对象名称完全一致；
sizeBytes 是 OSS 中最终 gzip 对象的大小；
uncompressedSizeBytes 是 gzip 解压后内容的大小；
recordCount 与解压后的有效 JSONL 行数一致；
totalFileCount 等于 files 数组长度；
totalRecordCount 等于所有分片 recordCount 之和；
SHA-256 对上传到 OSS 的 gzip 文件字节计算，而不是对解压后的 JSONL 计算；
如同时使用 OSS CRC64，应在上传后核对 CRC64，但 CRC64 不替代协议中的 SHA-256。

Manifest 一旦上传成功，该批次的数据文件和 Manifest 均不得修改。如果内容需要纠正，必须生成更大的新 batchVersion 并发布新批次。

四、业务级滚动 Meta
4.1 作用

每个 businessCode 在业务根目录维护一份 meta.json。它保存保留期内所有已完整发布批次的索引，防止消费任务因维护、暂停或故障数小时后只能看到最新批次而漏数。

Meta 是批次发现入口，不直接重复保存数据文件清单。完整文件信息以每个批次自己的 Manifest 为准。

4.2 格式示例




{
  "protocolVersion": "1.0",
  "businessCode": "book-online-sources",
  "generatedAt": "2026-09-15T14:00:00Z",
  "retentionHours": 4320,
  "firstBatchVersion": "20260915100000123-0001",
  "latestBatchVersion": "20260915103000123-0001",
  "batches": [
    {
      "batchVersion": "20260915100000123-0001",
      "producedAt": "2026-09-15T10:00:00Z",
      "manifestPath": "batches/20260915100000123-0001/manifest.json"
    },
    {
      "batchVersion": "20260915103000123-0001",
      "producedAt": "2026-09-15T10:30:00Z",
      "manifestPath": "batches/20260915103000123-0001/manifest.json"
    }
  ]
}

4.3 字段说明
字段	类型	必填	说明
protocolVersion	string	是	当前固定为 1.0
businessCode	string	是	业务唯一标识，必须与 OSS 业务目录一致
generatedAt	string	是	本次 Meta 生成时间，ISO 8601 UTC 格式
retentionHours	integer	是	Meta 批次列表的声明保留时长
firstBatchVersion	string	是	当前 batches 中最小的批次版本
latestBatchVersion	string	是	当前 batches 中最大的批次版本
batches	array	是	保留期内已完整发布的批次，按 batchVersion 严格升序排列

batches 条目：

字段	类型	必填	说明
batchVersion	string	是	已发布批次版本
producedAt	string	是	对应 Manifest 中的批次生产完成时间
manifestPath	string	是	相对于当前业务根目录的 Manifest 路径
4.4 保留规则
meta.json 建议至少保留最近 4320 小时，即 180 天的批次；
数据文件和 Manifest 建议至少保留 180 天，且不得短于 Meta 的保留时间；
清理 Meta 条目时以 producedAt 为准；
刚超过保留期的条目可以在下一次 Meta 更新时清理；
删除 Meta 中的旧条目不代表立即删除对应批次对象；
如果业务的最长可能停机时间超过 180 天，应相应增加 retentionHours；
如果消费方反馈存在未消费的旧批次，生产方应暂停清理相关批次对象。
4.5 更新规则
Meta 只能包含 Manifest 已成功上传的批次；
更新时必须先读取当前 Meta，在原有未过期列表上追加新批次；
不得仅写入最新批次并丢弃保留期内的历史批次；
使用一次完整的 OSS PutObject 覆盖 meta.json，不得分段追加；
新 Meta 上传成功后才表示新批次对消费方可发现；
更新失败时应重试，重试不得创建重复批次条目；
同一个 batchVersion 在 batches 中只能出现一次；
多个生产实例可能同时更新同一 Meta 时，必须使用单写者、分布式锁，或基于 ETag 的条件写入，防止后写入者覆盖先写入者的新批次；
使用 ETag 条件写入失败时，应重新读取最新 Meta、合并新批次后再重试。
五、生产方上传流程
5.1 标准发布流程

生产方必须严格按照以下顺序发布一个批次：

生成批次版本。 为当前 businessCode 生成严格大于历史批次的唯一 batchVersion。
生成业务记录。 按本协议输出 upsert、delete 业务变更记录，完成格式和字段校验。
排序与去重。 按约定业务键稳定排序；如业务允许，合并同一个 id 在当前批次内的重复变更。
切分并压缩。 按完整记录边界生成独立的 data_partNNNN.jsonl.gz，确保每个 gzip 对象最终不超过 10 MiB。
计算统计信息。 记录每个分片的压缩大小、解压大小、记录数以及 gzip 文件 SHA-256。
上传所有数据分片。 上传到 batches/{batchVersion}/，此时不要更新 Meta。
校验 OSS 对象。 确认全部分片存在，OSS 对象大小与本地结果一致；如启用 OSS CRC64，同时完成 CRC64 校验。
生成 Manifest。 使用最终上传结果生成完整的 manifest.json，再次校验文件数和记录总数。
上传 Manifest。 将 Manifest 上传到当前批次目录。Manifest 成功上传后，该批次内容即被冻结，不得再修改。
读取并合并 Meta。 读取当前 meta.json，保留所有未过期批次并追加当前批次，按版本升序排序。
发布新 Meta。 使用完整 PutObject 原子覆盖 meta.json；存在并发生产者时使用条件写入或锁。
记录发布结果。 保存本次批次版本、文件数量、总记录数、Manifest 路径和 Meta 更新结果，用于监控与审计。

发布顺序可概括为：



生成数据
  → 上传全部 gzip 分片
  → 校验 OSS 分片
  → 上传不可变 Manifest
  → 更新滚动 Meta
  → 批次对消费方可见


禁止使用以下顺序：



先更新 Meta → 再上传 Manifest 或数据文件


否则消费方可能发现尚未上传完整的批次。

5.2 失败处理
失败阶段	处理方式
数据生成、压缩或校验失败	终止发布；不上传 Manifest；不更新 Meta
部分数据分片上传失败	重试失败分片；全部成功前不上传 Manifest
数据分片已上传，但 Manifest 上传失败	重试上传同一份 Manifest；不更新 Meta
Manifest 已上传，但 Meta 更新失败	保持数据和 Manifest 不变，只重试 Meta 合并与更新
Meta 更新结果不确定	重新读取 Meta；已有当前批次则视为成功，否则执行幂等重试
发布后发现内容错误	不得覆盖原对象；生成更大的新批次进行修正

没有进入 Meta 的批次不会被正常消费流程发现。生产方可以定期扫描“存在 Manifest 但未进入 Meta”的批次并告警或补登记，但不得自动删除仍可能处于发布过程中的对象。

5.3 发布前检查清单
businessCode、OSS 目录和 Meta 一一对应；
batchVersion 唯一、严格递增且可按字符串排序；
所有 JSONL 行都是合法的 UTF-8 JSON 对象；
op、id、data 符合操作语义；
每个 gzip 文件均可独立解压且不超过 10 MiB；
分片编号从 0000 开始且连续；
Manifest 中的文件大小、记录数和 SHA-256 均来自最终 gzip 文件；
Manifest 的总文件数和总记录数计算正确；
已确认所有数据分片上传成功后才上传 Manifest；
已确认 Manifest 上传成功后才更新 Meta；
Meta 保留期覆盖约定的最长消费停机时间；
Meta 更新保留了所有尚未过期的历史批次；
已处理多个生产实例并发更新 Meta 的覆盖风险。
附录 A：最小完整示例
A.1 OSS 对象




/prod/book-online-sources/meta.json
/prod/book-online-sources/batches/20260915100000123-0001/manifest.json
/prod/book-online-sources/batches/20260915100000123-0001/data_part0000.jsonl.gz

A.2 数据文件解压后的内容




{"op":"upsert","id":"uuid-1","data":{"bookSourceId":"uuid-1","siteCode":"qidian","title":"星辰变"}}
{"op":"delete","id":"uuid-2"}

A.3 Manifest




{
  "protocolVersion": "1.0",
  "businessCode": "book-online-sources",
  "batchVersion": "20260915100000123-0001",
  "batchType": "incremental",
  "producedAt": "2026-09-15T10:00:00Z",
  "format": "jsonl",
  "compression": "gzip",
  "contentEncoding": "utf-8",
  "files": [
    {
      "sequence": 0,
      "fileName": "data_part0000.jsonl.gz",
      "sizeBytes": 356,
      "uncompressedSizeBytes": 212,
      "recordCount": 2,
      "checksum": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ],
  "totalFileCount": 1,
  "totalRecordCount": 2
}

A.4 滚动 Meta




{
  "protocolVersion": "1.0",
  "businessCode": "book-online-sources",
  "generatedAt": "2026-09-15T10:00:05Z",
  "retentionHours": 4320,
  "firstBatchVersion": "20260915100000123-0001",
  "latestBatchVersion": "20260915100000123-0001",
  "batches": [
    {
      "batchVersion": "20260915100000123-0001",
      "producedAt": "2026-09-15T10:00:00Z",
      "manifestPath": "batches/20260915100000123-0001/manifest.json"
    }
  ]
}



你目前 OSS 里的大文件：
data/full.jsonl                    约 89.6 MiB
data/jsonl/20260928.jsonl          约 85.1 MiB
data/jsonl/20261013.jsonl          约 4.7 MiB
data/picture/...jpg
不能直接按文档发布，因为协议要求：
- JSONL 必须 gzip 压缩
- 每个 gzip 压缩后不超过 10 MiB
- 文件名必须为 data_partNNNN.jsonl.gz
- 先上传所有分片，再上传 Manifest，最后更新 Meta
建议新数据使用以下对象路径。不要手工把 / 写成 %2F，boto3 会负责 URL 编码。
prod/research-report-images/meta.json

prod/research-report-images/batches/20261026090000000-0001/
├── data_part0000.jsonl.gz
├── data_part0001.jsonl.gz
├── data_part0002.jsonl.gz
└── manifest.json
manifest.json
假设该批次有两个 gzip 分片：
{
  "protocolVersion": "1.0",
  "businessCode": "research-report-images",
  "batchVersion": "20261026090000000-0001",
  "batchType": "incremental",
  "producedAt": "2026-10-26T09:00:00Z",
  "format": "jsonl",
  "compression": "gzip",
  "contentEncoding": "utf-8",
  "files": [
    {
      "sequence": 0,
      "fileName": "data_part0000.jsonl.gz",
      "sizeBytes": 8380123,
      "uncompressedSizeBytes": 18743291,
      "recordCount": 30000,
      "checksum": "sha256:实际压缩文件的64位小写SHA256"
    },
    {
      "sequence": 1,
      "fileName": "data_part0001.jsonl.gz",
      "sizeBytes": 4157821,
      "uncompressedSizeBytes": 9342280,
      "recordCount": 15000,
      "checksum": "sha256:实际压缩文件的64位小写SHA256"
    }
  ],
  "totalFileCount": 2,
  "totalRecordCount": 45000
}
其中：
- sizeBytes：.gz 文件实际大小
- uncompressedSizeBytes：解压后 JSONL 的字节数
- recordCount：JSONL 行数
- checksum：对 .gz 文件计算 SHA-256
- JPG 图片不要直接放进该 files 数组；协议目前只定义 JSONL 分片。JSON记录中可以保存图片的 Object Key
meta.json
第一个批次发布时：
{
  "protocolVersion": "1.0",
  "businessCode": "research-report-images",
  "generatedAt": "2026-10-26T09:00:05Z",
  "retentionHours": 4320,
  "firstBatchVersion": "20261026090000000-0001",
  "latestBatchVersion": "20261026090000000-0001",
  "batches": [
    {
      "batchVersion": "20261026090000000-0001",
      "producedAt": "2026-10-26T09:00:00Z",
      "manifestPath": "batches/20261026090000000-0001/manifest.json"
    }
  ]
}
后续新增批次时，不能只保留最新批次，而要追加：
{
  "protocolVersion": "1.0",
  "businessCode": "research-report-images",
  "generatedAt": "2026-10-26T10:00:05Z",
  "retentionHours": 4320,
  "firstBatchVersion": "20261026090000000-0001",
  "latestBatchVersion": "20261026100000000-0001",
  "batches": [
    {
      "batchVersion": "20261026090000000-0001",
      "producedAt": "2026-10-26T09:00:00Z",
      "manifestPath": "batches/20261026090000000-0001/manifest.json"
    },
    {
      "batchVersion": "20261026100000000-0001",
      "producedAt": "2026-10-26T10:00:00Z",
      "manifestPath": "batches/20261026100000000-0001/manifest.json"
    }
  ]
}