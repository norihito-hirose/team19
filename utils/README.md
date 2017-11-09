SRC Downloader from Github
=======================================================

# Requirements

- go [[install instruction](https://golang.org/doc/install)]
- wget

# Build command line app `create_repo_list`

```bash
$ cd $TEAM19/utils
$ make
```
*note* `$TEAM19` replace your location this repository root path.

# Usage

## Authorization Github

Create Github [Personal API tokens](https://github.com/blog/1509-personal-api-tokens).

set created API token to environment variable `GITHUB_TOKEN`.

```bash
$ export GITHUB_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

## Create repo list by topics (ex: python+tensorflow)

create list of URL that downloadable archived master source code.

```bash
$ cd $TEAM19
$ ./bin/create_repo_list -q SEARCH_QUERY > OUTPUT.urllist
```

`SEARCH_QUERY` is github search query. Please show [Search repositories](https://developer.github.com/v3/search/#search-repositories) the `q` search terms.

ex:
```bash
$ ./bin/create_repo_list -q "language:python topic:tensorflow" > python_tf_repos.urllist
```

## Download master.zip from created url list

```bash
$ wget -i OUTPUT.urllist -P DOWNLOAD_DIR
```
*note* need empty DOWNLOAD_DIR.

## Unarchive all master.zip

```bash
$ find DOWNLOAD_DIR -name "*" -type f -exec unzip -n -d SRC_DIR {} \;
```

## Delete not python files

```bash
$ find SRC_DIR -not -name "*.py" -type f -exec rm {} \;
```
