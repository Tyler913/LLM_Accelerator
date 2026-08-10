#ifndef TEST_NETIF_XADAPTER_H
#define TEST_NETIF_XADAPTER_H

struct netif {
    int marker;
};

int xemacif_input(struct netif *network_interface);

#endif /* TEST_NETIF_XADAPTER_H */
