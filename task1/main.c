#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct
{
    int id;
    char name[32];
    double price;
    int tax_flag;
} Product;

typedef struct
{
    int product_id;
    int qty;
} CartItem;

typedef struct
{
    int product_index;
    int qty;
    double unit_price;
    int tax_flag;
    double discount_rate;
    double tax_rate;
    double discounted;
    double tax;
    double line_total;
} ThreadData;

static Product PRODUCTS[] = {
    {1, "Bread", 2.50, 1},
    {2, "Milk", 3.20, 0},
    {3, "Eggs", 4.75, 0},
    {4, "Coffee", 9.99, 1},
    {5, "Sugar", 2.10, 1},
    {6, "Apple", 5.40, 0}};

static int PRODUCTS_COUNT = sizeof(PRODUCTS) / sizeof(PRODUCTS[0]);

void print_products(void)
{
    printf("Available Products:\n ");
    for (int i = 0; i < PRODUCTS_COUNT; i++)
    {
        printf("%d. %s $%.2f\n",
               PRODUCTS[i].id,
               PRODUCTS[i].name,
               PRODUCTS[i].price);
    }
}

int find_product_index_by_id(int id)
{
    for (int i = 0; i < PRODUCTS_COUNT; i++)
    {
        if (PRODUCTS[i].id == id)
            return i;
    }
    return -1;
}

void *compute_discount_tax(void *arg)
{
    ThreadData *td = (ThreadData *)arg;
    double subtotal = td->unit_price * td->qty;
    td->discounted = subtotal * (1.0 - td->discount_rate);
    td->tax = td->tax_flag ? td->discounted * td->tax_rate : 0.0;
    td->line_total = td->discounted + td->tax;
    pthread_exit(NULL);
}

int main(void)
{
    const double TAX_RATE = 0.13;
    const double DISCOUNT_RATE = 0.15;
    CartItem cart[128];
    int cart_count = 0;
    print_products();
    while (1)
    {
        printf("\nOrder? (y/n): ");
        char choice = 0;
        if (scanf(" %c", &choice) != 1)
        {
            printf("Invalid input.\n");
            return 1;
        }
        if (choice == 'n' || choice == 'N')
        {
            break;
        }
        if (choice != 'y' && choice != 'Y')
        {
            printf("Please enter y or n.\n");
            continue;
        }
        printf("Enter product ID: ");
        int pid = 0;
        if (scanf(" %d", &pid) != 1)
        {
            printf("Invalid product ID.\n");
            return 1;
        }
        int idx = find_product_index_by_id(pid);
        if (idx < 0)
        {
            printf("Product not found.\n");
            continue;
        }
        printf("Enter quantity: ");
        int qty = 0;
        if (scanf(" %d", &qty) != 1 || qty <= 0)
        {
            printf("Invalid quantity.\n");
            continue;
        }
        if (cart_count >= (int)(sizeof(cart) / sizeof(cart[0])))
        {
            printf("Cart is full.\n");
            break;
        }
        cart[cart_count].product_id = pid;
        cart[cart_count].qty = qty;
        cart_count++;
    }

    if (cart_count == 0)
    {
        printf("\nNo items ordered. Goodbye.\n");
        return 0;
    }

    ThreadData *thread_data = malloc(cart_count * sizeof(ThreadData));
    pthread_t *threads = malloc(cart_count * sizeof(pthread_t));

    for (int i = 0; i < cart_count; i++)
    {
        int idx = find_product_index_by_id(cart[i].product_id);
        if (idx < 0)
            continue;
        thread_data[i].product_index = idx;
        thread_data[i].qty = cart[i].qty;
        thread_data[i].unit_price = PRODUCTS[idx].price;
        thread_data[i].tax_flag = PRODUCTS[idx].tax_flag;
        thread_data[i].discount_rate = DISCOUNT_RATE;
        thread_data[i].tax_rate = TAX_RATE;
        pthread_create(&threads[i], NULL, compute_discount_tax, &thread_data[i]);
    }

    for (int i = 0; i < cart_count; i++)
    {
        pthread_join(threads[i], NULL);
    }

    printf("\nYour Cart:\n");
    printf("ID Name Qty Unit Subtotal Tax Line Total\n");
    double grand_total = 0.0;
    for (int i = 0; i < cart_count; i++)
    {
        int idx = thread_data[i].product_index;
        if (idx < 0)
            continue;
        grand_total += thread_data[i].line_total;
        printf("%d %s %d %.2f %.2f %.2f %.2f\n",
               PRODUCTS[idx].id,
               PRODUCTS[idx].name,
               thread_data[i].qty,
               thread_data[i].unit_price,
               thread_data[i].discounted,
               thread_data[i].tax,
               thread_data[i].line_total);
    }

    printf("\nGrand Total: %.2f\n", grand_total);
    printf("Thank you. Goodbye.\n");

    free(thread_data);
    free(threads);
    return 0;
}
