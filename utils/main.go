package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/google/go-github/github"
	"golang.org/x/oauth2"
)

func main() {
	query := flag.String("q", "language:python topic:tensorflow", "Search query for github repository.")
	sort := flag.String("s", "stars", "sort. default stars")
	order := flag.String("o", "desc", "order. desc, asc")

	flag.Parse()
	ctx := context.Background()

	ts := oauth2.StaticTokenSource(
		&oauth2.Token{AccessToken: os.Getenv("GITHUB_TOKEN")},
	)
	tc := oauth2.NewClient(ctx, ts)

	c := github.NewClient(tc)

	opts := &github.SearchOptions{
		Sort:  *sort,
		Order: *order,
		ListOptions: github.ListOptions{
			PerPage: 100,
		},
	}

	log.Println("Start search repos. q:", *query)

	var allRepos []*github.RepositoriesSearchResult
	for {
		repos, resp, err := c.Search.Repositories(ctx, *query, opts)

		// check rate limit
		if _, ok := err.(*github.RateLimitError); ok {
			log.Println("hit rate limit.")
			log.Println("waiting 60 sec.")
			time.Sleep(60 * time.Second)
			continue
		}

		if repos.GetIncompleteResults() {
			log.Println("retry incomplete result.")
			continue
		}

		log.Println("downloaded search results #", opts.Page)
		allRepos = append(allRepos, repos)

		if resp.NextPage == 0 {
			log.Println("not exists next page. last page num #", opts.Page)
			break
		}

		opts.Page = resp.NextPage
	}

	log.Println("downloaded page num: ", len(allRepos))
	log.Println("total repos: ", allRepos[0].GetTotal())

	// stdout repo urls
	for _, v := range allRepos {
		for _, r := range v.Repositories {
			fmt.Printf("%s/archive/master.zip\n", *r.HTMLURL)
		}
	}
}
