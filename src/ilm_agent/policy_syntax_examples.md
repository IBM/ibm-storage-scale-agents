# IBM Storage Scale ILM Policy Syntax Reference

Reference: https://www.ibm.com/docs/en/storage-scale/6.0.0?topic=rules-policy-syntax

## Important: Filesystem Context

**CRITICAL**: When a user mentions a filesystem (e.g., "filesystem fs1"), they are specifying which storage filesystem to operate on, NOT a path filter. The filesystem context is determined by where the policy is applied, not by adding PATH_NAME conditions.

**DO NOT** add `PATH_NAME LIKE '/filesystem_name/%'` unless the user explicitly requests filtering by a specific directory path within the filesystem.


## File Migration Rule Syntax

```
RULE ['RuleName'] [WHEN TimeBooleanExpression]
  MIGRATE
    [COMPRESS ({'yes' | 'no' | 'z' | 'lz4' | 'zfast' | 'alphae' | 'alphah'})]
    [FROM POOL 'FromPoolName']
    [THRESHOLD (HighPercentage[,LowPercentage[,PremigratePercentage]])]
    [WEIGHT (WeightExpression)]
  TO POOL 'ToPoolName'    
    [LIMIT (OccupancyPercentage)]
    [REPLICATE (DataReplication)]
    [FOR FILESET ('FilesetName'[,'FilesetName']...)]
    [SHOW (['String'] SqlExpression)]
    [SIZE (numeric-sql-expression)]
    [ACTION (SqlExpression)]
    [WHERE SqlExpression]
```

## File Deletion Rule Syntax

```
RULE ['RuleName'] [WHEN TimeBooleanExpression]
  DELETE
    [DIRECTORIES_PLUS]
    [FROM POOL 'FromPoolName']
    [THRESHOLD (HighPercentage[,LowPercentage])]
    [WEIGHT (WeightExpression)]
    [FOR FILESET ('FilesetName'[,'FilesetName']...)]
    [SHOW (['String'] SqlExpression)]
    [SIZE (numeric-sql-expression)]
    [ACTION (SqlExpression)]
    [WHERE SqlExpression]
```

## Common SQL Expressions for WHERE Clause

### File Age (Access Time)
```
DAYS(CURRENT_TIMESTAMP) - DAYS(ACCESS_TIME) > N
```

### File Age (Modification Time)
```
DAYS(CURRENT_TIMESTAMP) - DAYS(MODIFICATION_TIME) > N
```

### File Age (Creation Time)
```
DAYS(CURRENT_TIMESTAMP) - DAYS(CREATION_TIME) > N
```

### File Size
```
KB_ALLOCATED > N
MB_ALLOCATED > N
GB_ALLOCATED > N
```

### File Name Pattern (case-insensitive)
```
lower(NAME) LIKE '%.ext'
```

### Path Pattern
```
PATH_NAME LIKE '%/directory/%'
```

### Combining Conditions
```
(condition1) AND (condition2)
(condition1) OR (condition2)
```

## Complete Working Examples

### Example 1: Migrate files older than 30 days
```
RULE 'migrate_old_files' MIGRATE TO POOL 'archive' WHERE DAYS(CURRENT_TIMESTAMP) - DAYS(ACCESS_TIME) > 30
```

### Example 2: Migrate large files (>1GB = 1048576 KB)
```
RULE 'migrate_large_files' MIGRATE TO POOL 'capacity' WHERE KB_ALLOCATED > 1048576
```

### Example 3: Migrate files older than 90 days to cold storage
```
RULE 'migrate_to_cold' MIGRATE TO POOL 'cold_storage' WHERE DAYS(CURRENT_TIMESTAMP) - DAYS(ACCESS_TIME) > 90
```

### Example 4: Migrate log files older than 7 days
```
RULE 'migrate_old_logs' MIGRATE TO POOL 'archive' WHERE (DAYS(CURRENT_TIMESTAMP) - DAYS(ACCESS_TIME) > 7) AND (lower(NAME) LIKE '%.log')
```

### Example 5: Migrate video files larger than 10GB (10485760 KB)
```
RULE 'migrate_large_videos' MIGRATE TO POOL 'capacity' WHERE (KB_ALLOCATED > 10485760) AND (lower(NAME) LIKE '%.mp4' OR lower(NAME) LIKE '%.avi' OR lower(NAME) LIKE '%.mov')
```

### Example 6: Delete temporary files older than 30 days from /tmp directory
```
RULE 'delete_temp' DELETE FROM POOL 'system' WHERE (DAYS(CURRENT_TIMESTAMP) - DAYS(ACCESS_TIME) > 30) AND (lower(NAME) LIKE '%.tmp' OR PATH_NAME LIKE '%/tmp/%')
```

### Example 7: Migrate files from specific directory path
```
RULE 'migrate_archive_dir' MIGRATE TO POOL 'archive' WHERE PATH_NAME LIKE '%/archive/%'
```

### Example 8: Migrate files not accessed in 180 days and larger than 100MB (102400 KB)
```
RULE 'migrate_cold_large' MIGRATE TO POOL 'cold_storage' WHERE (DAYS(CURRENT_TIMESTAMP) - DAYS(ACCESS_TIME) > 180) AND (KB_ALLOCATED > 102400)
```

### Example 9: Delete files older than 1 year from temp pool
```
RULE 'delete_old_temp' DELETE FROM POOL 'temp' WHERE DAYS(CURRENT_TIMESTAMP) - DAYS(CREATION_TIME) > 365
```

### Example 10: Migrate backup files older than 60 days
```
RULE 'migrate_old_backups' MIGRATE TO POOL 'backup_archive' WHERE (DAYS(CURRENT_TIMESTAMP) - DAYS(MODIFICATION_TIME) > 60) AND (lower(NAME) LIKE '%.bak' OR lower(NAME) LIKE '%.backup')
```

### Example 11: Migrate from system pool to archive pool
```
RULE 'migrate_from_system' MIGRATE FROM POOL 'system' TO POOL 'archive' WHERE DAYS(CURRENT_TIMESTAMP) - DAYS(ACCESS_TIME) > 30
```

### Example 12: Migrate with compression
```
RULE 'migrate_compress' MIGRATE COMPRESS ('lz4') TO POOL 'archive' WHERE DAYS(CURRENT_TIMESTAMP) - DAYS(ACCESS_TIME) > 60
```

## Syntax Requirements

Rule names must be enclosed in single quotes: RULE 'my_rule'
Pool names must be enclosed in single quotes: TO POOL 'archive'
String literals use single quotes: '%.log'
Use lower() function for case-insensitive matching: lower(NAME) LIKE '%.txt'
Age calculation uses DAYS() function: DAYS(CURRENT_TIMESTAMP) - DAYS(ACCESS_TIME) > 30
File size uses KB_ALLOCATED, MB_ALLOCATED, or GB_ALLOCATED: KB_ALLOCATED > 1048576
