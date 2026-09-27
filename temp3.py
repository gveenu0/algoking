import requests
from bs4 import BeautifulSoup



def fetch_document(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def parse_points(html: str) -> list[tuple[int, int, str]]:
    soup = BeautifulSoup(html, "html.parser")
    points = []
    for row in soup.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) != 3:
            continue
        try:
            points.append(
                (
                    int(cols[0].get_text(strip=True)),
                    int(cols[2].get_text(strip=True)),
                    cols[1].get_text(strip=True),
                )
            )
        except ValueError:
            continue
    return points

def build_grid(points: list[tuple[int, int, str]]) -> list[list[str]]:
    max_x = max(x for x, _, _ in points)
    max_y = max(y for _, y, _ in points)
    grid = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    for x, y, char in points:
        grid[y][x] = char
    return grid

def print_grid(grid: list[list[str]]) -> None:
    for row in reversed(grid):
        print("".join(row))

def print_secret_message(url: str) -> None:
    html = fetch_document(url)
    points = parse_points(html)

    if not points:
        print("No valid data found.")
        return

    grid = build_grid(points)
    print_grid(grid)


url = " https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"
print_secret_message(url)